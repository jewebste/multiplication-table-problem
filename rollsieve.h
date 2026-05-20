#ifndef _ROLLSIEVE
#define _ROLLSIEVE

/*
Incrementally finds primes using the rolling sieve

Rollsieve r(start);     -- constructor, will enumerate primes > start
bool r.next();          -- true if next integer is prime; advances to next integer
uint64_t r.nextprime(); -- returns the next larger prime
uint64_t r.getn();      -- returns the current value of n

To list all primes from 1 to n, use this:

 Rollsieve r(1);
 for(uint64_t p=r.nextprime(); p<=n; p=r.nextprime()) { std::cout << p << std::endl; }

Credit: Jonathan Sorenson
*/


#include<cmath>
#include<cstdint>
#include<vector>


//using namespace std;

class Factorlist2
{
    public:
    static const int maxplen=15;
    uint64_t list;
    char plen;
    char ptr[maxplen];

    Factorlist2(): list(0), plen(0) {}

    inline void clear() { list=0; plen=0; }

    uint32_t get(uint32_t pos);

    uint32_t getbitlen(uint32_t pos);

    uint32_t bitlength(uint32_t x);
    void add(uint32_t p, uint32_t bitlen);

    inline void add(uint32_t p) { add(p,bitlength(p)); }
    inline void push(uint32_t p, int bitlen) { add(p,bitlen); }
    inline void push(uint32_t p) { add(p); }
    inline uint32_t pop() { return get( --plen ); }
    inline bool isempty() { return (plen==0); }
    inline uint32_t gettop() { return get(plen-1); }
    inline uint32_t gettopbitlen() { return getbitlen(plen-1); }
    inline void makeempty() { clear(); }

    void getlist(uint64_t n, std::vector<uint64_t> & plist);

};  // end of Factorlist2 class

class Rollsieve
{

    static const int primeslen = 168;
    static const uint16_t primesmax = 1000;
    static const uint16_t rollsieve_primes[168]; // initialized at the end of the file

    std::vector<Factorlist2> T;
    uint32_t delta;

    uint64_t n, r;
    uint32_t pos;

    public:

    uint64_t s;

    Rollsieve(uint64_t start);

    inline uint64_t getn() { return n; }

    bool next();  // this code is nearly verbatim from the paper

    uint64_t nextprime() { while(!next()); return n-1; }

    bool isnextprime();

    void getlist(std::vector<uint64_t> & plist) { T[pos].getlist(n,plist); }

    const uint16_t primes[168] = {
     2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
    31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
   127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
   179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
   233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
   283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
   353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
   419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
   467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
   547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
   607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
   661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
   739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
   811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
   877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
   947, 953, 967, 971, 977, 983, 991, 997
  };
};



#endif
