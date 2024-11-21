from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np


class PathSorting:
    def __init__(self, epsilon=None, method="fast_non_dominated_sort") -> None:
        super().__init__()
        self.epsilon = epsilon
        self.method = method
        '''self.eu_tol = eu_tol
        self.x0 = curr
        self.xF = currf'''

    def do(self, X, F ,anch_val,counts,cnst ,x0 ,xF, return_rank=False, only_non_dominated_front=False, n_stop_if_ranked=None, **kwargs):
        if n_stop_if_ranked is None:
            n_stop_if_ranked = int(1e8)
        fronts_new = []
        n_ranked   = 0
        Anchors = []
        cd_path    = [np.inf]
        prev       = x0
        prevf      = xF[0,:]
        index = 1
        dom_mat = np.zeros((F.shape[0], F.shape[0]))
        flag = False
        for i in range(dom_mat.shape[0] - 1):
            for j in range(i + 1, dom_mat.shape[0]):
                if F[j, 0] < F[i, 0] and F[j, 1] < F[i, 1]:
                    dom_mat[i, j] = 1
                elif F[i, 0] < F[j, 0] and F[i, 1] < F[j, 1]:
                    dom_mat[j, i] = 1

        while n_ranked < n_stop_if_ranked:  # or (len(Anchors) + len(B_T)) < F.shape[0]:
            front = np.where(np.sum(dom_mat, axis=1, keepdims=True) == 0)[0]
            front = list(set(front) - set(Anchors))

            if len(front) == 0:
                break

            non_values = F[front, :]
            first_ind = non_values[:, 0].argsort()[::-1]
            temp = F[[front[ind] for ind in first_ind]]
            non_ind = temp[:, -1].argsort()
            points = [front[first_ind[ind]] for ind in non_ind]
            indices = []

            if len(Anchors) == 0:
                Anchors.append(front[first_ind[non_ind[0]]])
                if len(counts) == 0:
                    flag = True
                    counts.append(0)
                cd_path.append(0)
                n_ranked += 1

            for i in range(len(points)):
                val_I = X[points[i], :].T
                I = F[points[i], :]

                if I[1] == 0:
                    continue
                SCV,diff,_ = cnst.calculate(prev,prevf,val_I,I)


                if  diff == 0:
                    Anchors.append(points[i])
                    if index < len(anch_val) and flag == False:
                        change = anch_val[index, 1] - I[1]
                        if change <= 1e-4 and change >= 0:
                            if anch_val[index, 0] - I[0] <= 1e-4 and anch_val[index, 0] - I[0] >= 0:
                                counts[index - 1] += 1
                            else:
                                counts[index - 1] = 0
                                flag = True
                            # counts[index-1] += 1
                        else:
                            counts[index - 1] = 0
                            flag = True
                    elif flag == True and index < len(anch_val):
                        counts[index - 1] = 0
                    else:
                        counts.append(0)
                    index += 1
                    prev = val_I
                    prevf = I
                    cd_path.append(SCV)
                    n_ranked += 1

                else:
                    indices.append(points[i])

                if n_ranked >= n_stop_if_ranked:
                    break

            for ele in indices:
                dom_mat[:, ele] = 0
                dom_mat[ele, :] = 2

        fronts_new.append(np.array(Anchors))

        B_T = [i for i in range(F.shape[0]) if i not in Anchors]
        Rest = NonDominatedSorting(self.epsilon, self.method).do(F[B_T])

        if len(counts) > len(Anchors):
            counts = counts[:len(Anchors)]
            counts[-1] = 0

        if n_ranked < n_stop_if_ranked:
            for front in Rest:
                act_ind = []
                for ind in front:
                    act_ind.append(B_T[ind])
                fronts_new.append(np.array(act_ind))
                n_ranked += front.shape[0]
                if n_ranked >= n_stop_if_ranked:
                    break


        if only_non_dominated_front:
            return fronts_new[0] ,cd_path

        if return_rank:
            rank = rank_from_fronts(fronts_new, F.shape[0])
            return fronts_new, rank ,cd_path


        return fronts_new ,cd_path,counts


def rank_from_fronts(fronts, n):
    # create the rank array and set values
    rank = np.full(n, 1e16, dtype=int)
    for i, front in enumerate(fronts):
        rank[front] = i

    return rank