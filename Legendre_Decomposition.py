import numpy as np

class legendre_decomposition:
    """
    A 2D implementation of lengendre decomposition.
    
    Sugiyama, M., Nakahara, H., Tsuda, K.:
    Legendre Decomposition for Tensors,
    NeurIPS 2018
    """
    def __init__(self, P):
        """
        Parameters:
        -----------
        P: numpy.ndarray
            Input Matrix
        
        Returns:
        --------
        None
        """
        self.P = P
        self.theta_mat = np.zeros(P.shape)
        self.eta_emp_mat = self._compute_eta(P)

    def reconstruct(self):
        """
        Use theta to reconstruct input
        
        Paramaters:
        -----------
        None
        
        Returns:
        --------
        None
        """
        exp_theta_mat = np.exp(self.theta_mat)
        k, l = exp_theta_mat.shape
        Q = np.empty((k, l))
        for i in range(k):
            for j in range(l):
                Q[i, j] = np.prod(exp_theta_mat[np.arange(0, i+1)][:, np.arange(0, j+1)])
        psi = np.sum(Q)
        Q /= psi
        return Q

    def _compute_eta(self, P_mat):
        """
        Compute eta for parameter P_mat
        
        Parameters:
        -----------
        P_mat: numpy.ndarray
            Normalised matrix.
        
        Returns:
        --------
        eta_mat: numpy.ndarray
            eta matrix for P_mat.
        """
        k, l = P_mat.shape
        eta_mat = np.empty((k, l))
        for i in range(k):
            for j in range(l):
                eta_mat[i,j] = np.sum(P_mat[np.arange(i, k)][:, np.arange(j, l)])
        return eta_mat

    def _gradient_descent_step(self, lr=0.01, verbose=False):
        """
        Single step to train the algorithm. Minimises the KL Divergence
        between the input and the model.
        
        Paramaters:
        -----------
        lr: int
            Learning rate
        verbose: bool
            print progress of training
            
        Returns:
        --------
        None
        """
        gradient = self._compute_eta(self.reconstruct()) - self.eta_emp_mat
        
        if verbose:
            print('============================================')
            print('eta:\n', np.around(self._compute_eta(self.reconstruct()), 3))
            print('eta_emp:\n', np.around(self.eta_emp_mat, 3))
            print('gradient:\n', np.around(gradient, 3))
            print('theta:\n', np.around(self.theta_mat, 3))
            print('np.exp(theta):\n', np.around(np.exp(self.theta_mat), 3))
            print('P:\n', np.around(self.P, 3))
            print('Q:\n', np.around(self.reconstruct(), 3))
            print('total error:', np.around(np.sum(gradient**2)**0.5, 3))
            print('============================================')
        
        self.theta_mat -= lr*gradient

    def train(self, N_iter, lr=0.01, verbose=False, verbose_step=100):
        """
        Trains the model.
        
        Paramaters:
        -----------
        lr: int
            Learning rate
        verbose: bool
            print progress of training
        verbose_step: int
            Number of iterations before printing progress of training.
            verbose must be set to True to print progress.

        Returns:
        --------
        None
        """
        for i in range(N_iter):
            if i % verbose_step == 0:
                verbose_flag = verbose
            else:
                verbose_flag = False
            self._gradient_descent_step(lr=lr, verbose=verbose_flag)

    def get_theta(self):
        """
        Returns theta, the natural parameter.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        self.theta_mat: numpy.ndarray
            Returns a matrix of theta, the natural parameter.
        """
        return self.theta_mat

    def get_eta(self):
        """
        Returns eta, the expectation parameter.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        eta_mat: numpy.ndarray
            Returns a matrix of eta, the expectation parameter.
        """
        eta_mat = self._compute_eta(self.reconstruct())
        return eta_mat

    def change_P(self, P):
        """
        Parameters:
        -----------
        P: numpy.ndarray
            The input matrix
        
        Returns:
        --------
        None
        """
        self.P = P
        self.eta_est_mat = self._compute_eta(P)

def main():
	np.random.seed(1)
	P = np.random.rand(5,5)
	P /= np.sum(P)
	ld = legendre_decomposition(P)
	ld.train(10000, lr=1, verbose=True, verbose_step=1000)

if __name__ == '__main__':
	main()