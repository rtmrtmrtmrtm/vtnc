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

class Decoder {
private:
  Source *_src;
  WindowSource *_ws;
  FFTCache *_fftc;
  FindLeader *_fl;

public:
  Decoder() : _src(0), _ws(0), _fftc(0), _fl(0) { }
  ~Decoder() {
    if(_src)
      delete _src;
    if(_ws)
      delete _ws;
    if(_fftc)
      delete _fftc;
    if(_fl)
      delete _fl;
  }

  void go() {
    _src = Wav::open("ardop1.wav");
    // _src = new Leader(1500 + 0);
    assert(_src->rate() == RATE);
    _ws = new WindowSource(_src, 10 * 240 + 512);
    _fftc = new FFTCache(_ws);
    _fl = new FindLeader(_fftc);

    int quarter = 240 / 4;
    for(long long off = 0; ; off += quarter) {
      double hz, score;
      bool ok = _fl->analyze(off, hz, score);
      if(ok){
        double next_hz, next_score;
        bool next_ok = _fl->analyze(off + quarter, next_hz, next_score);
        if(next_ok == false || score > next_score){
          // this is it!
          //printf("%lld best: %.1f %.1f\n", off, hz, score);
          int type;
          off = decode_type(off + 240, hz, type);
          if(type >= 0x31 && type < 0x38){
            // CONREQxxx: 14 bytes of payload
            std::vector<int> bytes = decode_4fsk(off, hz, 14*4);
            printf("payload: ");
            for(int i = 0; i < bytes.size(); i++){
              printf("%02x ", bytes[i]);
            }
            printf("\n");
          }
        }
      }
    }
  }

  // 10 50-baud 4FSK symbols.
  // 8 bits type, 8 bits type^sessionID, 2 symbols of parity.
  // off is where we start.
  // hz is nominally 1500.
  // tones are 1425, 1475, 1525, and 1575.
  // so we can't directly use 240-point FFT.
  long long decode_type(long long off, double hz, int &type) {
    std::vector<int> bytes = decode_4fsk(off, hz, 10);
    type = bytes[0];
    printf("%lld: type %02x\n", off, type);
    return off + 10*240;
  }

  // decode the payload of a 4fsk 50 baud packet, e.g. a CONREQxxx.
  // we expect n_symbols total.
  // returns an array of decoded bytes.
  std::vector<int> decode_4fsk(long long off, double hz, int n_symbols) {
    int symbols[n_symbols];
    for(int i = 0; i < n_symbols; i++){
      std::vector<double> v = _ws->readat(off, 240);
      v.insert(v.begin(), 120, 0.0);
      v.insert(v.end(), 120, 0.0);
      std::vector<complex> ff = fft(v); // 480-point fft, so 25-hz bins
      int maxsym = -1;
      double maxmag = 0;
      int sym = 0;
      for(int bin = 1425/25; bin <= 1575/25; bin += 2){
        double mag = magnitude(ff[bin]) + magnitude(ff[bin+1]); // 50 hz bin
        if(maxsym < 0 || mag > maxmag){
          maxsym = sym;
          maxmag = mag;
        }
        sym++;
      }
      symbols[i] = maxsym;
      off += 240;
    }

    // no convert 2-bit symbols to bytes.
    // most-significant symbol first.
    std::vector<int> bytes;
    int i = 0;
    while(i < n_symbols){
      int x = 0;
      for(int j = 0; j < 4 && i < n_symbols; j++){
        x |= symbols[i] << (6 - (j*2));
        i++;
      }
      bytes.push_back(x);
    }

    return bytes;
  }
};

int
main()
{
  Decoder d;
  d.go();
}
