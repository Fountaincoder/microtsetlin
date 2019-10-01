#PYTHON VERSION > 3
import numpy as np
import itertools
import numba

LIT,NOT_LIT   = 0,1 #LIT = postive literal
REWARD,INACTION,PENALTY = 0,1,2
NO_ACT, ACT             = 0,1

B = 3
L = 1

u = [L,-L,L,-L,L,-L,-B,B,-L,L,-L,L,-L,L,B,-B,-L,L,-L,L,-L,L,B,-B,L,-L,L,-L,L,-L,-B,B]
u2 =[L,-L,L,-L,L, 0,-B,B, 0,0, 0,0,-L,0,0, 0, 0,0, 0,0,-L,0,0, 0,L,-L,L,-L,L,-L,-B,B]

def calc_all_clause_outputs(x_hat, tsetlin, prune = False):
    """Calculate all clauses using the fact that any FALSE in an AND statement terminates
    an AND.
    """
    num_c, num_f, _ = tsetlin.shape
    def calc_clause_output(c):
        """output is a tuple (x,bool), where bool is whether the clause has used literals"""
        used = 0
        for f in range(num_f):
            used += action(tsetlin[c,f,LIT]) + action(tsetlin[c,f,NOT_LIT])
            if (action(tsetlin[c,f,LIT]) == ACT and x_hat[f] == 0) or (action(tsetlin[c,f,NOT_LIT]) == ACT and x_hat[f] == 1):
                return (0,True)
        live = True if not prune else (used != 0)
        return (1, live)
    return [calc_clause_output(c) for c in range(num_c)]

@numba.jit(target='cpu')
def update(t_in,y_hat,polarity,clause_output,x_hat,tsetlin,T = 15):
    """ Perform tsetlin array update."""
    num_c, num_f, _ = tsetlin.shape
    y_sft= y_hat << 4
    for c in range(num_c):
        c_out, live  = clause_output[c]
        p_sft = polarity[c] << 3
        c_sft = c_out << 2
        for f in range(num_f):
            for s in [LIT,NOT_LIT]:
                x_d = x_hat[f] if s == LIT else (1 - x_hat[f])
                x_sft = x_d << 1
                up = u2[y_sft|p_sft|c_sft|x_sft|action(tsetlin[c,f,s])]
                if up != 0:
                    if action(tsetlin[c,f,s]) == NO_ACT:
                        tsetlin[c,f,s] -= up
                    else:
                        tsetlin[c,f,s] += up
                    if tsetlin[c,f,s] > MAXSTATE : tsetlin[c,f,s] = MAXSTATE
                    if tsetlin[c,f,s] < MINSTATE : tsetlin[c,f,s] = MINSTATE

X   = [[0,0],[0,1],[1,0],[1,1]]
Y   = [0,1,1,0]
r   = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
rc  = itertools.cycle(r)
rc2 = itertools.cycle(r)

num_features  = len(X[0])
num_clauses   = 10  #must be even
num_c2        = num_clauses//2
MAXSTATE      = 20 #must be even
MINSTATE      = 1
MIDSTATE      = MAXSTATE//2
action        = lambda state: NO_ACT if state <= MIDSTATE else ACT #note: MIDSTATE = INACTION
tsetlin       = np.random.choice([float(MIDSTATE), float(MIDSTATE+1)], size=(num_clauses,num_features, len([LIT,NOT_LIT])))
clause_sign   = np.array(num_c2*[1] + num_c2*[-1])
polarity      = np.array(num_c2*[1] + num_c2*[0] )
iterations    = 300

def train():
    for i in range(iterations):
        e = 0
        out = []
        for x_hat,y_hat in zip(X,Y):
            clause_output = calc_all_clause_outputs(x_hat, tsetlin)
            tot_c_outputs = np.sum([c_out*sign if live else 0 for (c_out,live),sign in zip(clause_output,clause_sign)])
            out += [tot_c_outputs]
            y_est = 1 if tot_c_outputs >= 0 else 0 #nte: total 0 = output 1
            e += 1 if y_est != y_hat else 0

            update(tot_c_outputs,y_hat,polarity,clause_output,x_hat,tsetlin,T=4)
        print("it", i, e,out)
train()
