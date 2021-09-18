/* Individual.h
This file contains the Individual class,
should not be modified.

Authors
Derek N.A. Alves, 
Bruno C.S. Nogueira, 
Davi R.C Oliveira and
Ermeson C. Andrade.

Instituto de Computação, Universidade Federal de Alagoas.
Maceió, Alagoas, Brasil.
*/
 
#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include <utility>
#include <vector>

class Individual{
    public:
    float* aleles;
    std::pair< float, int > fitness;

    Individual(float* al, std::pair<float, int> ft);
    ~Individual();
    void ToString();
};

Individual::Individual(float* al, std::pair<float, int> ft)
{
    aleles = al;
    fitness = ft;
}

Individual::~Individual(){
    free(aleles);
}

void Individual::ToString()
{
    std::cout << "Individual: " << fitness.second << "Fitness: " << fitness.first << std::endl << "Aleles: ";
    //for(int i = 0; i < 128; i++) std::cout << aleles[i] << " ";
    std::cout << std::endl;
}

#endif
