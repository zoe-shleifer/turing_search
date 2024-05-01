import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from joblib import Parallel, delayed
# update in time
delta_t = 1.0


# simulation steps
N_simulation_steps = 10000
def get_initial_configuration(random_influence, N):
        N = N
        A = (1-random_influence) * np.ones((N,N)) + random_influence * np.random.random((N,N))
        
        # Let's assume there's only a bit of B everywhere
        B = random_influence * np.random.random((N,N))
        
        # Now let's add a disturbance in the center
        N2 = N//2
        radius = r = int(N/10.0)
        
        A[N2-r:N2+r, N2-r:N2+r] = 0.50
        B[N2-r:N2+r, N2-r:N2+r] = 0.25
        assert(not np.isnan(A[0][0]))
        print("initialized.")
        return A,B

init_A, init_B = get_initial_configuration(0.2, 200)

class gs:
    def __init__(self, DA, DB, f, k, im_path=None, ax = None) -> None:
          self.A = np.copy(init_A)
          self.B =  np.copy(init_B)
          self.delta_t = delta_t
          self.DA = DA
          self.DB = DB
          self.f = f
          self.im_path = im_path
          self.ax = ax
          self.k= k
          self.N = 200
          self.N_steps = 10000
          self.random_influence = 0.2
    def discrete_laplacian(self, M):
        """Get the discrete Laplacian of matrix M"""
        L = -4*M
        L += np.roll(M, (0,-1), (0,1)) # right neighbor
        L += np.roll(M, (0,+1), (0,1)) # left neighbor
        L += np.roll(M, (-1,0), (0,1)) # top neighbor
        L += np.roll(M, (+1,0), (0,1)) # bottom neighbor
        
        return L
    def draw(self):
        """draw the concentrations"""
        if(not self.ax):
            fig, self.ax = plt.subplots(1,1,figsize=(4,4))
        self.ax.imshow(self.B, cmap='Greys')
        self.ax.axis('off')
        if(not self.im_path):
            plt.show()
        else:
            plt.savefig(self.im_path)
    def gray_scott_update(self, A, B, DA, DB, f, k, delta_t):
        # Let's get the discrete Laplacians first
            LA = self.discrete_laplacian(A)
            LB = self.discrete_laplacian(B)

            # Now apply the update formula
            diff_A = (DA*LA - A*B**2 + f*(1-A)) * delta_t
            diff_B = (DB*LB + A*B**2 - (k+f)*B) * delta_t

            A += diff_A
            B += diff_B
            return A,B
    def make_pattern(self):
        #print(f"DA: {self.DA}, DB: {self.DB}, f:{self.f}, k:{self.k}, delta_t: {delta_t}\n")
        A, B = np.copy(init_A), np.copy(init_B)
        #print(A[0,10:20])
        for t in range(N_simulation_steps):
            A, B = self.gray_scott_update(A, B, self.DA, self.DB, self.f, self.k, delta_t)
            test = A[0].sum()
            if((test>199.9999 and test<200) or np.isnan(test)):
                 break
        #print(A[0,10:20])
        self.A, self.B = A,B
from time import time
from joblib import Parallel, delayed

def slow_function(gs_obj):
        gs_obj.make_pattern()
        return gs_obj.A

def run_threads(params):
        nthreads = params.shape[1]
        objs = [gs(*(params[:,i])) for i in range(nthreads)]
        return Parallel(n_jobs=6)(delayed(slow_function)(objs[i]) for i in range(len(objs)))
if __name__ == "__main__":
    DA_values = np.linspace(0.12, 0.18, num=20)
    DB_values = np.linspace(0.06, 0.12, num=20)
    f_values = np.linspace(0.05, 0.06, num=20)
    k_values = np.linspace(0.055, 0.065, num=20)

    nthreads = 1000
    params = np.zeros((4, nthreads))
    params[0, :] = np.random.choice(DA_values, nthreads)  # Choose randomly from DA_values
    params[1, :] = np.random.choice(DB_values, nthreads)  # Choose randomly from DB_values
    params[2, :] = np.random.choice(f_values, nthreads)   # Choose randomly from f_value
    params[3, :] = np.random.choice(k_values, nthreads)   # Choose randomly from k_values
    init_A, init_B = get_initial_configuration(0.2, 200)
    # simulates waiting time (e.g., an API call/response)
    # secs = time()
    # A_arr = run_threads(params)
    # print("%.4f" % (time() - secs))
    
    # bad_params = []
    # j = 0
    # for i in range(nthreads):
    #     if((A_arr[i][0].sum()>199.9999 and A_arr[i][0].sum()<200) or np.isnan(A_arr[i]).sum()>0):
    #         bad_params.append(i)
    # good_params = [i for i in range(nthreads) if i not in bad_params]
    # for i in good_params:
    #     plt.imshow(A_arr[i], cmap = "Greys")
    #     plt.axis('off')
    #     t = [ "%.3f_"%i for i in params[:,i]]
    #     plt.savefig("pats/"+"".join(t)+".png", bbox_inches=0)
    # good_params = [i for i in range(nthreads) if i not in bad_params]
    fig, axs = plt.subplots(2, 1, figsize = (8,4))
    axs = axs.flatten()
    axs[0].imshow(init_A, cmap="Reds")
    axs[0].axis('off')
    axs[0].set_title('initial concentration of U')
    axs[1].imshow(init_B, cmap="Blues")
    axs[1].set_title('initial concentration of V')
    axs[1].axis('off')
    # axs[0].scatter(params[0, bad_params], params[1, bad_params], color = "blue")
    # axs[0].scatter(params[0, good_params], params[1, good_params], color = "red")
    # axs[1].scatter(params[2, bad_params], params[3, bad_params], color = "blue")
    # axs[1].scatter(params[2, good_params], params[3, good_params], color = "red")
    plt.savefig("ititial_same.png")