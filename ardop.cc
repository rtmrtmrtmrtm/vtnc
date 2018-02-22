#include <stdio.h>
#include <stdlib.h>
#include <sndfile.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <algorithm>
#include <vector>
#include <complex>

#define RATE 12000

const double PI  = 3.141592653589793238463;

typedef std::complex<double> complex;

class Source {
public:
  Source() { }
  virtual ~Source() { }
  virtual int rate() = 0;
  virtual std::vector<double> read() = 0;
};

class Wav : public Source {
private:
  int _rate;
  int _channels;
  SNDFILE *_sf;

public:
  Wav() : _sf(0) { }
  ~Wav() { if(_sf != 0) sf_close(_sf); }

  static Wav *open(const char *filename) {
    SF_INFO sfi;
    SNDFILE *sf = sf_open(filename, SFM_READ, &sfi);
    if(sf != 0){
      assert(sfi.channels == 1);
      Wav *w = new Wav();
      w->_rate = sfi.samplerate;
      w->_channels = sfi.channels;
      w->_sf = sf;
      return w;
    } else {
      return 0;
    }
  }

  int rate() { return _rate; }

  std::vector<double> read() {
    std::vector<double> v(512);
    int nn = sf_read_double(_sf, v.data(), v.size());
    v.resize(nn);
    return v;
  }
};

// phase-shifting leader.
// it is not clear this is what the real ARDOP does, though.
// its pulses are shaped, probably sinc.
// and they don't seem to invert phase at minimum amplitude.
class Leader : public Source {
private:
  double _hz; // nominally 1500 Hz
  long long _off ; // next sample number to generate

public:
  Leader(double hz) : _off(0), _hz(hz) { }

  int rate() { return RATE; }

  std::vector<double> read() {
    std::vector<double> v(100);
    for(int i = 0; i < v.size(); i++){
      v[i] = one();
    }
    return v;
  }

  double one() {
    int baud = 50;
    int baudlen = rate() / baud;
    double phase = (_off / (rate() / _hz)) * 2 * PI;

    // invert phase every symbol.
    phase += PI * ((_off / baudlen) % 2);

    // but not the 12th symbol -- so as to produce the "sync" symbol.
    if((_off / baudlen) == 11)
      phase += PI;

    // shape pulse with sinc window
    double x = PI * (((_off % baudlen) - (baudlen / 2.0)) / (baudlen / 2.0)); // -pi .. pi
    double ampl;
    if(x == 0.0){
      ampl = 1.0;
    } else {
      ampl = sin(x) / x;
    }
    
    double y = cos(phase) * ampl;

    _off++;
    return y;
  }
};

class BlockUp : public Source {
private:
  int _blocksize;
  std::vector<double> _buf;
  Source *_src;
  
public:
  BlockUp(Source *src, int blocksize) : _src(src), _blocksize(blocksize) { }
  ~BlockUp() { }
  int rate() { return _src->rate(); }
  std::vector<double> read() {
    while(1){
      if(_buf.size() >= _blocksize){
        std::vector<double> v(_buf.begin(), _buf.begin() + _blocksize);
        _buf.erase(_buf.begin(), _buf.begin() + _blocksize);
        return v;
      }
      std::vector<double> v = _src->read();
      if(v.size() == 0)
        return std::vector<double>();
      _buf.insert(_buf.end(), v.begin(), v.end());
    }
  }
};

// cache the last "window" samples.
class WindowSource : public Source {
private:
  int _window; // buffer a window of this many samples in the past.
  std::vector<double> _buf;
  long long _off; // first sample beyond _buf.
  Source *_src;

public:
  WindowSource(Source *src, int window) : _src(src), _window(window), _off(0) { }
  ~WindowSource() { }
  int rate() { return _src->rate(); }
  std::vector<double> read() {
    std::vector<double> v = _src->read();
    _buf.insert(_buf.end(), v.begin(), v.end());
    if(_buf.size() > _window){
      int n = _buf.size() - _window;
      _buf.erase(_buf.begin(), _buf.begin() + n);
    }
    _off += v.size();
    return v;
  }
  std::vector<double> readat(long long off, int n) {
    while(_off < off + n){
      read();
    }
    long long i0 = off - (_off - _buf.size());
    long long i1 = i0 + n;
    assert(i0 >= 0);
    assert(i1 <= _buf.size());
    std::vector<double> v(_buf.begin() + i0, _buf.begin() + i1);
    return v;
  }
};

// fft of one 50-baud symbol at 12000 samples/second.
// caller allocates and frees in and out.
// in[240]
// out[240/2+1] i.e. out[121]
std::vector<complex>
fft(std::vector<double> in)
{
  int fft_size = (in.size() / 2) + 1;
  double *inx = fftw_alloc_real(in.size());
  fftw_complex *outx = fftw_alloc_complex(fft_size);
  for(int i = 0; i < in.size(); i++)
    inx[i] = in[i];

  fftw_plan p;
  p = fftw_plan_dft_r2c_1d(in.size(), inx, outx, FFTW_ESTIMATE);
  fftw_execute(p);

  std::vector<complex> out(fft_size);
  for(int i = 0; i < fft_size; i++)
    out[i] = complex(outx[i][0], outx[i][1]); // real, imag

  fftw_destroy_plan(p);
  fftw_free(inx);
  fftw_free(outx);

  return out;
}

class FFTCache {
private:
  WindowSource *_src;

public:
  FFTCache(WindowSource *src) : _src(src) { }
  ~FFTCache() { }

  std::vector<complex> fft(long long off, int n) {
    return ::fft(_src->readat(off, n));
  }
};

double
phase(complex c)
{
  return atan2(c.imag(), c.real());
}

double
magnitude(complex c)
{
  return sqrt(c.imag()*c.imag() + c.real()*c.real());
}

// normalize to 0..2pi
double
twopi(double x)
{
  while(x < 0.0)
    x += 2*PI;
  while(x > 2*PI)
    x -= 2*PI;
  return x;
}

// look for a plausible leader.
class FindLeader {
private:
  static const int baud = 50;
  static const int symbol_samples = RATE / baud;
  static const int fft_size = (symbol_samples / 1) + 1;
  FFTCache *_fft;
  
public:
  FindLeader(FFTCache *fft) : _fft(fft) { }
  ~FindLeader() { }

  // off is the start of a putative sync symbol (at end of leader).
  bool analyze(long long off, double &best_hz, double &best_score) {
    if(off < symbol_samples*8)
      return false;
    
    best_score = 0.0; // higher is better
    best_hz = -1;
    int center_bin = 1500 / baud;
    for(int bin = center_bin-1; bin <= center_bin+1; bin++){
      double mag[8];
      double diff[8];
      // i=0 is sync symbol, i=1 is one before that, &c.
      for(int i = 0; i < 8; i++){
        std::vector<complex> prev = _fft->fft(off - (i+1)*symbol_samples, symbol_samples);
        std::vector<complex> me = _fft->fft(off - i*symbol_samples, symbol_samples);
        double prev_ph = phase(prev[bin]);
        double me_ph = phase(me[bin]);
        diff[i] = twopi(me_ph - prev_ph);
        mag[i] = magnitude(me[bin]);
      }

      // guess the frequency offset from the center of the FFT bin,
      // based on rate of phase change.
      double rate = 0.0;
      for(int i = 1; i < 8; i++){
        // diff[i] should be PI.
        rate += diff[i] - PI;
      }
      rate /= 7.0;
      double hzoff = ((rate / (2 * PI)) / symbol_samples) * RATE;

      double score = 0.0;
      double d_adj = twopi(diff[0] - rate);
      if(d_adj < 0.3 || d_adj > 2*PI-0.3){
        // most recent symbol looks like a sync (no phase inversion).
        // XXX 0.3 is arbitrary.

        // calculate score from phase inversion of symbols
        // before sync.
        bool ok = true;
        for(int i = 1; i < 8; i++){
          double d_adj = twopi(diff[i] - rate);
          score += mag[i] * (PI - abs(d_adj - PI));
          if(abs(d_adj - PI) > 0.3){
            // XXX 0.3 is arbitrary.
            ok = false;
          }
        }
        if(ok == false){
          score = 0.0;
        }
      }

      if(score > 0.0 && (best_hz < 0 || score > best_score)){
        best_score = score;
        best_hz = (baud * bin) + hzoff;
      }
    }
    if(best_score > 0.0){
      return true;
    } else {
      return false;
    }
  }
};

void
writetxt(std::vector<double> v, const char *filename)
{
  FILE *fp = fopen(filename, "w");
  for(int i = 0; i < v.size(); i++)
    fprintf(fp, "%f\n", v[i]);
  fclose(fp);
}

int
main(int argc, char *argv[])
{
  //Source *w = Wav::open("ardop1.wav");
  Source *w = new Leader(1500 + 0);
  assert(w);
  assert(w->rate() == RATE);
  WindowSource *ws = new WindowSource(w, 10 * 240);
  FFTCache *fftc = new FFTCache(ws);

  FindLeader *fl = new FindLeader(fftc);

  for(long long off = 0; ; off += 240/4) {
    double xhz;
    double xscore;
    bool ok = fl->analyze(off, xhz, xscore);
    if(ok){
      printf("%lld best: %.1f %.1f\n", off, xhz, xscore);
    }
  }

  delete fl;
  delete fftc;
  delete ws;
  delete w;
  
  return 0;
}
