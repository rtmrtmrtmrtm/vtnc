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
  long long _seq ; // next sample number to generate

public:
  Leader(double hz) : _seq(0), _hz(hz) { }

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
    double phase = (_seq / (rate() / _hz)) * 2 * PI;

    // invert phase every symbol.
    phase += PI * ((_seq / baudlen) % 2);

    // but not the 11th symbol -- so as to produce the "sync" symbol.
    if((_seq / baudlen) == 11)
      phase += PI;

    // shape pulse with sinc window
    double x = PI * (((_seq % baudlen) - (baudlen / 2.0)) / (baudlen / 2.0)); // -pi .. pi
    double ampl;
    if(x == 0.0){
      ampl = 1.0;
    } else {
      ampl = sin(x) / x;
    }
    
    double y = cos(phase) * ampl;

    _seq++;
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
// XXX perhaps create one FindLeader per quarter symbol time.
class FindLeader {
private:
  static const int baud = 50;
  static const int symbol_samples = RATE / baud;
  static const int fft_size = (symbol_samples / 1) + 1;
  std::vector<double> _buf; // accumulate one symbol's samples

  std::vector<std::vector<complex> > _window;
  int _wi;

  long long _seq; // seen this many samples
  
public:
  FindLeader() : _seq(0) {
    reset();
  }
  ~FindLeader() {
  }

  void reset() {
    _window.resize(0);
    _window.resize(10);
    for(int i = 0; i < _window.size(); i++)
      _window[i].resize(3);
    _buf.resize(0);
    _wi = 0;
  }
  
  void got(std::vector<double> a) {
    for(int i = 0; i < a.size(); i++){
      if(_buf.size() >= symbol_samples){
        digest();
        _buf.resize(0);
      }
      _buf.push_back(a[i]);
      _seq++;
    }
  }

  // called once per symbol time.
  void digest() {
    assert(_buf.size() == symbol_samples);
    std::vector<complex> trum = fft(_buf);

    int center_bin = 1500 / baud;
    for(int bin = 0; bin < 3; bin++){
      _window[_wi][bin] = trum[center_bin-1+bin];
    }
    _wi = (_wi + 1) % _window.size();
  }

  void analyze() {
    double best_score = 0.0; // higher is better
    double best_hz = -1;
    for(int bin = 0; bin < 3; bin++){
      double mag[8];
      double diff[8];
      // most recent symbol (candidate sync) in ph[0] &c.
      // _wi-1 is the most recent symbol.
      for(int i = 0; i < 8; i++){
        int i0 = (_wi - 2 - i + _window.size()) % _window.size(); // previous
        int i1 = (_wi - 1 - i + _window.size()) % _window.size(); // this one
        complex c0 = _window[i0][bin]; // previous
        complex c1 = _window[i1][bin]; // this one
        double ph0 = phase(c0);
        double ph1 = phase(c1);
        diff[i] = twopi(ph1 - ph0);
        mag[i] = magnitude(c1);
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
      double d0 = twopi(diff[0] - rate);
      if(d0 < 0.3 || d0 > 2*PI-0.3){
        // XXX 0.3 is arbitrary.
        // most recent symbol looks like a sync (no phase inversion).
        for(int i = 1; i < 8; i++){
          score += mag[i] * (PI - abs(diff[i] - PI));
        }
      }

      if(best_hz < 0 || score > best_score){
        best_score = score;
        best_hz = 1500 - 50 + (50 * bin) + hzoff;
      }
    }
    printf("%lld best: %.1f %.1f\n", _seq, best_hz, best_score);
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
  Source *w = new Leader(1500 + 41);
  assert(w);
  assert(w->rate() == RATE);
  Source *bu = new BlockUp(w, RATE / 50);

  // XXX maybe two of four of these, for different symbol alignments
  // in time.
  FindLeader *fl = new FindLeader();

  double aa[512];

  long long total = 0;
  while(1){
    std::vector<double> v = bu->read();
    //writetxt(v, "x");
    //exit(0);
    if(v.size() < 1)
      break;
    fl->got(v);
    total += v.size();
    // XXX need to call analyze() every symbol or quarter symbol.
    if(total >= 8*(RATE/50)){
      fl->analyze();
    }
  }

  delete bu;
  delete fl;
  delete w;
  
  return 0;
}
