import numpy

def functie_obiectiv(x, costuri): # x - individ (permutari - val intregi, unice, de la 1 la n), costuri - matricea de costuri
    n = len(x) #nr de gene
    fitness = 0 # un individ nu poate sa nu fie fezabil, dar poate sa fie (?)
    for i in range(n-1):
        fitness = fitness + costuri[x[i], x[i+1]] # din mat de costuri luam elem de pe poz i si i+1 (?)
    fitness = fitness + costuri[x[n-1], x[0]]
    return 1 / fitness  # ca sa fie o pb de maxim (maximizarea fitnessului - performanta mare=cost mic)



# -------------Seminar 5----------------
# primim ca parametru dimensiunea populatiei
def generare_populatie_initiala(dim):
    costuri = numpy.genfromtxt("costuri_cv.txt")
    n = len(costuri)  #nr de orase (cate gene va avea individul)
    populatie = []
    for i in range(dim):
        x = numpy.random.permutation(n)
        # fiind permutari nu mai facem fezabilitatea, dar facem calitatea (cat de bine imi rezolva problema; in cazul asta cautam costul ca de bun e)
        fitness = functie_obiectiv(x, costuri)
        x = list(x)
        x.append(fitness)
        populatie.append(x)
    return populatie



if __name__ == "__main__":
    populatie = generare_populatie_initiala(20)
    print("Populatia initiala: ")
    print(numpy.asarray(populatie))