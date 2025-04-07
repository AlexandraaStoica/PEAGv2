import numpy


def functie_obiectiv(x, fisier_costuri, fisier_valori, cost_max):
    cost = numpy.dot(x, fisier_costuri)
    fitness = numpy.dot(x, fisier_valori)
    return cost <= cost_max, fitness


def generare_populatie_initiala(dim, cost_max):
    c = numpy.genfromtxt("cost.txt")
    v = numpy.genfromtxt("valoare.txt")
    n = len(c)
    pop = []
    for i in range(dim):
        fezabil = False
        while fezabil == False:
            x = numpy.random.uniform(0, 1, n) # se schimba fata de RucsacDiscret
            fezabil, calitate = functie_obiectiv(x, c, v, cost_max)
        x = list(x)
        x.append(calitate)
        pop.append(x)
    return numpy.asarray(pop)


#------------ Adaugat in sem 4 -----------
def mutatie_gena(gena, sigma, a, b ): #sigma - deviatia standard cat sa se duca valorile zgomotului (atat neg, cat si poz) pe graficul Gauss || (a,b) - interval din care sa nu iasa valorile dupa adaugarea zgomotului
    zgomot = numpy.random.normal(0, sigma)
    gena_mutanta = gena + zgomot
    if gena_mutanta > b: # a = 0, b = 1
        gena_mutanta = b
    if gena_mutanta < a:
        gena_mutanta = a
    return gena_mutanta

def mutatie_populatie(populatie, cost_max, probabilitate_mutatie, sigma): #cost_max - pt ca o sa apelam functia_obiectiv
    c = numpy.genfromtxt("cost.txt")
    v = numpy.genfromtxt("valoare.txt")
    n = len(c) #nr de gene ale unui individ
    dim = len(populatie)  #nr de indivizi

    populatie_mutanta = populatie.copy()  #punem copia pop init in populatia_mutanta

    for i in range(dim):  #pentru fiecare individ din populatie
        x = populatie[i][:n].copy()  # :n - toate pana la n || extrag individul ca sa nu suprascriu populatia initiala
        for j in range (n):  # pentru fiecare gena
            r = numpy.random.uniform(0,1,1)    # generam in intervalul (0,1] 1 gena - generez "sansa" sa se intample sau nu mutatia
            if r < probabilitate_mutatie:
                x[j] = mutatie_gena(x[j], sigma, 0, 1)  # difera fata de pb RucsacDiscret
        fezabil, fitness = functie_obiectiv(x, c, v, cost_max)
        if fezabil == True: #facem cast fortat populatiei la lista pentru a putea folosi append si sa ne adauge la final fitness-ul
            x = list(x)
            x.append(fitness)
            populatie_mutanta[i] = x.copy()

    return numpy.asarray(populatie_mutanta) #facem array ca sa ne afiseze frumos populatia






# -------------Seminar 5 - recombinare-------------
# genele sunt valori reale si primeste ca parametru cei 2 parinti si valoarea arbitrara alfa subunitar neaparat
def crossover_singular(x1, x2, alpha):
    n = len(x1)
    # pozitia genei pe care se face incrucisarea
    index_incrucisare = numpy.random.randint(0, n)
    copil1 = x1.copy()
    copil2 = x2.copy()

    copil1[index_incrucisare] = alpha * x1[index_incrucisare] + (1 - alpha) * x2[index_incrucisare] #mergem in copilul 1 pe pozitia generata aleator anterior
    copil2[index_incrucisare] = alpha * x2[index_incrucisare] + (1 - alpha) * x1[index_incrucisare]

    return copil1, copil2
# TEMA - modificam aceasta functie pentru crossover simplu si total


def crossover_populatie(populatie, cost_max, probabilitate_de_crossover, alpha):
    c = numpy.genfromtxt("cost.txt")
    v = numpy.genfromtxt("valoare.txt")
    n = len(c)  # nr de gene ale unui individ
    dim = len(populatie)  # nr de indivizi
    populatie_copii = populatie.copy()

    for i in range(0, dim-1, 2): # mergem de la 0 la dim-1, in pas de 2 (din 2 in 2)
        x1 = populatie[i][:-1].copy() # selectam primele n coloane, cele crespunzatoare genelor
        x2 = populatie[i+1][:-1].copy()
        r = numpy.random.uniform(0,1)
        if r < probabilitate_de_crossover:
            copil1, copil2 = crossover_singular(x1, x2, alpha)  # difera fata de RucsacDiscret
            fezabil, fitness = functie_obiectiv(copil1, c, v, cost_max)
            if fezabil == True:  # daca copil 1 e fezabil
                populatie_copii[i][:-1] = copil1.copy()  # salvez genele coplului 1
                populatie_copii[i][-1] = fitness  # salvez fitness-ul copilului 1
            fezabil, fitness = functie_obiectiv(copil2, c, v, cost_max)
            if fezabil == True:  # daca copil 1 e fezabil
                populatie_copii[i][:-1] = copil2.copy()  # salvez genele coplului 2
                populatie_copii[i][-1] = fitness  # salvez fitness-ul copilului 2
    return populatie_copii



if __name__ == "__main__":
    populatie = generare_populatie_initiala(20, 50)
    print("Populatia initiala: ")
    print(populatie)

    print("Calitatea celui mai bun individ este: ")
    print(numpy.max(populatie[:, 15]))

    # print("-------------------------")
    # populatie_mutanta = mutatie_populatie(populatie, 50, 0.1, 0.05)  #zgomotul nu trebuie sa fie foarte mare
    # print("Populatia mutanta: ")

    # print(populatie)
    # print("Calitatea celui mai bun individ mutant este: ")
    # print(numpy.max(populatie_mutanta[:, 15]))

    # testare crossover:
    populatie_copii = crossover_populatie(populatie, 50, 0.8, 0.32)
    print("Populatia copii: ")
    print(populatie_copii)






# alternativa afisare rezultate
'''
if __name__ == "__main__":
    populatie = generare_populatie_initiala(20, 50)
    print("Populatia initiala: ")

    for i in range(20):
        print("")
        for j in range(16):
            print(f'{populatie[i, j]:.4f}', end=" ")

    print(" ")
    print("Calitatea celui mai bun individ este: ")
    print(f'{numpy.max(populatie[:, 15]):.4f}')
'''