
import numpy as np
from model.LocallySmoothedModel import LocallySmoothedModel
from rl_utilities import  sample_one_from_mdp
from tqdm import tqdm

class DDTD_PE:

    def __init__(self, mdp, policy, rank, alpha, model, update_Ehat_interval, use_eigen_E=True):
        self.mdp = mdp
        self.policy = policy
        self.rank = rank
        self.alpha = alpha
        self.use_eigen_E = use_eigen_E
        self.update_Ehat_interval = update_Ehat_interval
        self.model = model
        

    def train(self, num_iteration, lr_scheduler):
        discount = self.mdp.discount()
        num_actions = self.mdp.num_actions()
        num_states = self.mdp.num_states()


        E = np.zeros((num_states, num_states))
        translation_matrix = np.linalg.inv(np.identity(num_states) - self.alpha * discount * E)

        V = np.zeros((num_states))
        z = np.zeros((num_states))
        self.V_trace = np.zeros((num_iteration, num_states))


        with tqdm(iter(range(num_iteration)), desc=f"TTD{self.rank}", unit="itr", total=num_iteration) as outer_iters:
            for k in outer_iters:
                (s,a,r,next_s) = sample_one_from_mdp(self.mdp, self.policy)  
                self.model.update(np.array([(s, a, r, next_s)]))
                
                target = self.alpha * (r + discount * ( V[next_s] - ( E @ V )[s]) ) + (1 - self.alpha) * V[s]
                z[s] = z[s] + lr_scheduler.get_lr(current_iter=k) * (target - z[s])
                V =  translation_matrix @ z

                if k % self.update_Ehat_interval == 0:
                    E_dynamics = self.model.get_P_hat()[self.policy, np.arange(num_states), :]
                    if self.use_eigen_E:
                        eig_vals, right_vecs = self.find_eig(E_dynamics, self.rank)
                    else:
                        eig_vals, right_vecs = self.find_eig_orthogonal_iteration(E_dynamics, self.rank)
                    left_vecs = self.orthogonalize(right_vecs)
                    E = right_vecs @ np.diag(eig_vals) @ left_vecs.T
                    z = (np.identity(num_states) - self.alpha * discount * E) @ V
                    translation_matrix = np.linalg.inv(np.identity(num_states) - self.alpha * discount * E)
                self.V_trace[k, :] = V

    def find_eig(self, Ppi, rank):
        eigvals, eigvecs = np.linalg.eig(Ppi)
        for i in range(eigvecs.shape[1]):
            eigvecs[:, i] = eigvecs[:, i] / np.linalg.norm(eigvecs[:, i], ord=2)
        idx = np.abs(eigvals).argsort()[::-1]  
        return eigvals[idx[:rank]], eigvecs[:, idx[:rank]]

    def find_eig_orthogonal_iteration(self, Ppi, rank, num_iterations=200):
        # Initialize a random orthogonal matrix
        B = np.random.rand(Ppi.shape[0], rank)
        b= np.ones(Ppi.shape[0])
        b= b/np.linalg.norm(b, ord=2)
        B[:, 0] = b
        Q, _ = np.linalg.qr(B)

        for _ in range(num_iterations):
            # Perform one iteration of orthogonal iteration
            Z = np.dot(Ppi, Q)
            Q, _ = np.linalg.qr(Z)
            Q[:, 0] = b
        # Compute the eigenvalues and eigenvectors
        eigenvalues = np.diag(np.dot(Q.T, np.dot(Ppi, Q)))
        eigenvectors = Q

        for i in range(eigenvectors.shape[1]):
            eigenvectors[:, i] = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i], ord=2)

        return eigenvalues, eigenvectors

    def orthogonalize(self, right_eigvecs):
        num_vecs = right_eigvecs.shape[1]
        left_vecs = np.zeros_like(right_eigvecs, dtype=complex)
        # left_vecs = np.zeros_like(initial_left_vectors)
        for i in range(right_eigvecs.shape[1]):
            left_vec = right_eigvecs[:, i]
            if num_vecs > 1:
                other_right_vecs = np.delete(right_eigvecs, i, axis=1)
                Q, R = np.linalg.qr(other_right_vecs, mode="complete")
                # projection = other_right_vecs @ np.linalg.pinv(other_right_vecs) @ left_vec
                # left_vec = left_vec - projection
                candidates = Q[:, num_vecs:]

                left_vec = candidates @ candidates.T @ right_eigvecs[:, i]
            left_vec = left_vec / np.dot(left_vec, right_eigvecs[:, i])
            left_vecs[:, i] = left_vec

        return left_vecs
            

    def run(self, num_iteration, lr_scheduler, output_filename, save_to_file=True):
        self.train(num_iteration, lr_scheduler)
        if save_to_file:
            np.save(output_filename, self.V_trace)
        return self.V_trace


