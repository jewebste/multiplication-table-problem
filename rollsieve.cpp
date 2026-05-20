/*
Incrementally finds primes using the rolling sieve

Rollsieve r(start);     -- constructor, will enumerate primes > start
bool r.next();          -- true if next integer is prime; advances to next integer
uint64_t r.nextprime(); -- returns the next larger prime
uint64_t r.getn();      -- returns the current value of n

To list all primes from 1 to n, use this:

 Rollsieve r(1);
 for(uint64_t p=r.nextprime(); p<=n; p=r.nextprime()) { std::cout << p << std::endl; }

*/


#include "rollsieve.h"


//using namespace std;


inline uint32_t Factorlist2::get(uint32_t pos)
{ // assumes 0<=pos<plen
    uint32_t left;
    left= (pos>0)?ptr[pos-1]:0;
    uint64_t copy;
    copy=(list >> left);
    return (uint32_t)
    ((copy & ( (1ul<<(ptr[pos]-left) ) -1 ))<<1)|1ul;
}

inline uint32_t Factorlist2::getbitlen(uint32_t pos)
{ // assumes 0<=pos<plen
    uint32_t left;
    left= (pos>0)?ptr[pos-1]:0;
    return ptr[pos]-left+1;
}

uint32_t Factorlist2::bitlength(uint32_t x)
{
    uint32_t len=0;
    while((1ul<<len) < x) len++;
    return len;
}

inline void Factorlist2::add(uint32_t p, uint32_t bitlen)
{
    uint32_t pos=plen;
    plen++;
    uint32_t left= (pos>0)?ptr[pos-1]:0;
    uint64_t pbits=(p>>1);
    list = list | (pbits<<left);
    ptr[pos]=left+bitlen-1;
}

void Factorlist2::getlist(uint64_t n, std::vector<uint64_t> & plist)
{
    plist.clear();

    // is n even? 2 is not in list
    if((n&1ul)==0)
    {
        plist.push_back(2); n>>=1;
        while((n&1ul)==0) n>>=1;
    }

    uint32_t p;
    for(int i=0; i<plen; i++)
    {
        p=get(i);
        plist.push_back(p);
        while(n%p==0) n=n/p;
    }
    if(n>1) plist.push_back(n);
}

Rollsieve::Rollsieve(uint64_t start)
{
    if(start<2) start=2;
    uint32_t sqrtstart = (uint64_t) sqrtl((long double) start);
    r=sqrtstart+1; s=r*r;
    delta=r+2;
    pos=0; n=start;

    T.reserve(2*delta+1000); // some wiggle room
    T.resize(delta);
    for( uint32_t i = 0; i < delta; i++) T[i].makeempty();

    // small primes first - take from int.h array if possible
    // for(int i=0; i<primeslen && primes[i]<=sqrtstart; i++)
    for( uint32_t i=1; i<primeslen && primes[i]<=sqrtstart; i++)
    {
        uint32_t p=primes[i];
        uint32_t j=(p-(start%p))%p;
        T[j].push(p);
    }
    // now large primes
    if(sqrtstart>primesmax)
    {
        Rollsieve R(primesmax); // hooray for recursion
        uint32_t p=R.nextprime();
        while(p<=sqrtstart)
        {
            uint32_t j=(p-(start%p))%p;
            T[j].push(p);
            p=R.nextprime();
        }
    }
}


bool Rollsieve::next()  // this code is nearly verbatim from the paper
{
    bool isprime=(n%2!=0 || n==2);  // we no longer store 2
    while(!T[pos].isempty())
    {
        uint32_t bitlen=T[pos].gettopbitlen();
        uint32_t p=T[pos].pop();
        T[(pos+p)%delta].push(p,bitlen);
        isprime=false;
    }
    T[pos].clear();
    if(n==s)
    {
        if(isprime)
        {
            isprime=false;
            T[(pos+r)%delta].push(r);
        }
        r++; s=r*r;
    }
    n++; pos=(pos+1)%delta;
    if(pos==0)
    {
        delta+=2;
        T.resize(delta);
        T[delta-1].makeempty();
        T[delta-2].makeempty();
    }
    return isprime;
}


bool Rollsieve::isnextprime()
{
    // checks if n+1 is prime:
    // 1) n is even so that n+1 is odd
    // 2) T( pos + 1 ) needs to be empty
    // 3) n + 1 could be the square of the next unrecorded prime, check that it is not
    return ( (0 == n%2) && T[ (pos+1) % delta ].isempty() && ( s != ( n + 1 ) ) ) ;
}
