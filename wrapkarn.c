#include <math.h>
#include <stdio.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>
#include <assert.h>
#include "rs.h"

static void *rs;
static int first=1;

void rs_encode_(int *dgen, int *sent)
// Encode JT65 data dgen[12], producing sent[63].
{
  int dat1[12];
  int b[51];
  int i;

  if(first) {
    // Initialize the JT65 codec
    rs=init_rs_int(6,0x43,3,1,51,0);
    first=0;
  }

  // Reverse data order for the Karn codec.
  for(i=0; i<12; i++) {
    dat1[i]=dgen[11-i];
  }
  // Compute the parity symbols
  encode_rs_int(rs,dat1,b);

  // Move parity symbols and data into sent[] array, in reverse order.
  for (i = 0; i < 51; i++) sent[50-i] = b[i];
  for (i = 0; i < 12; i++) sent[i+51] = dat1[11-i];
}

void rs_decode_(int *recd0, int *era0, int *numera0, int *decoded, int *nerr)
// Decode JT65 received data recd0[63], producing decoded[12].
// Erasures are indicated in era0[numera].  The number of corrected
// errors is *nerr.  If the data are uncorrectable, *nerr=-1 is returned.
{
  int numera;
  int i;
  int era_pos[50];
  int recd[63];

  if(first) {
    rs=init_rs_int(6,0x43,3,1,51,0);
    first=0;
  }

  numera=*numera0;
  for(i=0; i<12; i++) recd[i]=recd0[62-i];
  for(i=0; i<51; i++) recd[12+i]=recd0[50-i];
  if(numera) 
    for(i=0; i<numera; i++) era_pos[i]=era0[i];
  *nerr=decode_rs_int(rs,recd,era_pos,numera);
  for(i=0; i<12; i++) decoded[i]=recd[11-i];
}


void rs_encode__(int *dgen, int *sent)
{
  rs_encode_(dgen, sent);
}

void rs_decode__(int *recd0, int *era0, int *numera0, int *decoded, int *nerr)
{
  rs_decode_(recd0, era0, numera0, decoded, nerr);
}

//
// new more-configurable Python interface.
//

static int nn; // number of symbols on the wire (63 for jt65a)
static int kk; // number of information symbols (12 for jt65a)

void
nrs_init(int symsize, int gfpoly, int fcr, int prim,
         int nroots, int pad){
  rs = init_rs_int(symsize, gfpoly, fcr, prim, nroots, pad);
  assert(rs != 0);
  // N = 2^symsize - pad - 1   --- number of symbols on the wire
  // K = N - nroots            --- number of information symbols
  nn = (1<<symsize) - pad - 1;
  kk = nn - nroots;
}

void
nrs_encode_(int *in, int *out)
// in[kk]
// out[nn]
{
  int inrev[1024];
  int enc[1024];
  int i;
  int nroots = nn - kk;

  assert(1024 >= kk);
  assert(1024 >= nn);

  // Reverse data order for the Karn codec.
  for(i=0; i<kk; i++) {
    inrev[i]=in[kk-1-i];
  }
  // Compute the parity symbols
  encode_rs_int(rs,inrev,enc);

  // Move parity symbols and data into out[] array, in reverse order.
  for (i = 0; i < nroots; i++) out[nroots-1-i] = enc[i];
  for (i = 0; i < kk; i++) out[i+nroots] = inrev[kk-1-i];
}

void
nrs_decode_(int *wire, int *era0, int *numera0, int *decoded, int *nerr)
// Decode JT65 received data wire[nn], producing decoded[kk].
// Erasures are indicated in era0[numera].  The number of corrected
// errors is *nerr.  If the data are uncorrectable, *nerr=-1 is returned.
{
  int numera;
  int i;
  int era_pos[1024];
  int recd[1024];
  int nroots = nn - kk;

  numera=*numera0;
  for(i=0; i<kk; i++) recd[i]=wire[nn-1-i];
  for(i=0; i<nroots; i++) recd[kk+i]=wire[nroots-1-i];
  if(numera) 
    for(i=0; i<numera; i++) era_pos[i]=era0[i];
  *nerr=decode_rs_int(rs,recd,era_pos,numera);
  for(i=0; i<kk; i++) decoded[i]=recd[kk-1-i];
}
