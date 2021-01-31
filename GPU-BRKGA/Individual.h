#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <utility>
#include <vector>


//Individual Class, represents an individual.
class Individual{
    public:
    float* aleles;
    std::pair< float, int > fitness;

    Individual(float* al, std::pair<float, int> ft);
    ~Individual();
};

Individual::Individual(float* al, std::pair<float, int> ft)
{
    aleles = al;
    fitness = ft;
}

Individual::~Individual(){}

#endif
