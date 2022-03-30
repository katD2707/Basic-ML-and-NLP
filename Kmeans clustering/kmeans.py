from mimetypes import init
import numpy as np
from collections import defaultdict
from email.policy import default


class Member:
    def __init__(self, r_d, label=None, doc_id=None):
        self.r_d = r_d
        self.label = label
        self.doc_id = doc_id
    
class Cluster:
    def __init__(self):
        self.centroid = None
        self.members = []

    def reset_members(self):
        self.members = []
    
    def add_members(self, member):
        self.members.append(member)

class Kmeans:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.clusters = [Cluster() for _ in range(self.num_clusters)]
        self.E = []     
        self.S = 0      

    def load_data(self, data_path):
        def sparse_to_dense(sparse_r_d, vocab_size):
            r_d = [0.0 for _ in range(vocab_size)]
            indices_tfidfs = sparse_r_d.split()
            for index_tfidf in indices_tfidfs:
                index = int(index_tfidf.split(':')[0])
                tfidf = float(index_tfidf.split(':')[1])
                r_d[index] = tfidf
            return np.array(r_d)

        with open(data_path) as f:
            d_lines = f.read().splitlines()
        with open(data_path + '/../' + 'word_idfs.txt') as f:
            vocab_size = len(f.read().splitlines())
        
        self.data = []
        self.label_count = defaultdict(int)
        for data_id, d in enumerate(d_lines):
            feature = d.split('<fff>')
            label, doc_id = int(feature[0]), int(feature[1])
            self.label_count[label] += 1
            r_d = sparse_to_dense(sparse_r_d=feature[2], vocab_size=vocab_size)

            self.data.append(Member(r_d=r_d, label=label, doc_id=doc_id))

    def random_init(self, seed_value="random"):    
        if seed_value == "kmeans++":
            pass
        else:                               
            rand_centroids = np.random.choice(self.data, size=self.num_clusters, replace=False)         
            
            for cluster_idx in range(self.num_clusters):
                self.clusters[cluster_idx].centroid = rand_centroids[cluster_idx].r_d
            
    def compute_similarity(self, member, centroid):
        return np.dot(member.r_d, centroid) /           \
                (np.linalg.norm(member.r_d, ord=2)*np.linalg.norm(centroid, ord=2))

    def select_clusters_for(self, member):
        best_fit_cluster = None
        max_similarity = -1
        for cluster in self.clusters:
            similarity = self.compute_similarity(member, cluster.centroid)
            if similarity > max_similarity:
                best_fit_cluster = cluster
                max_similarity = similarity
        
        best_fit_cluster.add_members(member)
        return max_similarity

    def update_centroid_of(self, cluster):
        member_r_ds = [member.r_d for member in cluster.members]
        aver_r_d = np.mean(member_r_ds, axis=0)
        # sqrt_sum_sqr = np.sqrt(np.sum(aver_r_d**2))
        # new_centroid = np.array([value/sqrt_sum_sqr for value in aver_r_d])       
                                                                                    
        # cluster.centroid = new_centroid                                                           
        cluster.centroid = aver_r_d

    def stopping_condition(self, criterion, threshold):
        criteria = ['centroid', 'similarity', 'max_iters']
        assert criterion in criteria
        if criterion == 'max_iters':
            if self.iteration >= threshold:
                return True
            else:
                return False
        elif criterion == 'centroid':
            E_new = [list(cluster.centroid) for cluster in self.clusters]
            E_new_minus_E = [centroid for centroid in E_new if centroid not in self.E]
            self.E = E_new
            if len(E_new_minus_E) <= threshold:
                return True
            else:
                return False
        else:
            new_S_minus_S = self.new_S - self.S
            self.S = self.new_S
            if new_S_minus_S <= threshold:
                return True
            else:
                return False

    def run(self, seed_value, criterion, threshold):
        self.random_init(seed_value)

        self.iteration = 0
        while True:
            for cluster in self.clusters:
                cluster.reset_members()
            self.new_S = 0
            for member in self.data:
                max_S = self.select_clusters_for(member)
                self.new_S += max_S
            for cluster in self.clusters:
                self.update_centroid_of(cluster)
            
            self.iteration += 1
            if self.stopping_condition(criterion, threshold):
                break


    def compute_purity(self):
        majority_sum = 0
        for cluster in self.clusters:
            member_labels = [member.label for member in cluster.members]
            max_count = max([member_labels.count(label) for label in range(self.num_clusters)])
            majority_sum += max_count

        return majority_sum * 1./len(self.data)

    def compute_NMI(self):
        I_value, H_omega, H_C, N = 0., 0., 0., len(self.data)
        
        for cluster in self.clusters:
            wk = len(cluster.members) * 1.
            H_omega +=  -wk/N * np.log10(wk/N)
            member_labels = [member.label for member in cluster.members]
            for label in range(20):
                wk_cj = member_labels.count(label)*1.
                cj = self.label_count[label]
                I_value += wk_cj/N * np.log10(N * wk_cj/(wk*cj) + 1e-12) #what 1e-12 for?
        
        for label in range(20):
            cj = self.label_count[label] * 1.
            H_C += -cj/N * np.log10(cj/N)

        return I_value * 2 / (H_omega + H_C)

def load_data(data_path):                       
    def sparse_to_dense(sparse_r_d, vocab_size):
        r_d = [0.0 for _ in range(vocab_size)]
        indices_tfidfs = sparse_r_d.split()
        for index_tfidf in indices_tfidfs:
            index = int(index_tfidf.split(':')[0])
            tfidf = float(index_tfidf.split(':')[1])
            r_d[index] = tfidf
        return np.array(r_d)

    with open(data_path) as f:
        d_lines = f.read().splitlines()
    with open(data_path + '/../' + 'word_idfs.txt') as f:
        vocab_size = len(f.read().splitlines())
        
    data = []
    labels = []

    for data_id, d in enumerate(d_lines):
        feature = d.split('<fff>')
        label, doc_id = int(feature[0]), int(feature[1])
        labels.append(label)
        r_d = sparse_to_dense(sparse_r_d=feature[2], vocab_size=vocab_size)

        data.append(r_d)

    return data, labels     

def compute_accuracy(predicted_y, expected_y):                              
    matches = np.equal(predicted_y, expected_y)
    accuracy = np.sum(matches.astype(float)) / len(expected_y)              

    return accuracy

def clustering_with_KMeans(data_path): 
    data, labels = load_data(data_path)

    from sklearn.cluster import KMeans
    from scipy.sparse import csr_matrix
    X = csr_matrix(data)
    print('=========')
    kmeans = KMeans(
        n_clusters=20,
        init='random',
        n_init=5,
        tol=1e-3,
        random_state=2018
    ).fit(X)
    
    label = kmeans.labels_
  

def classifying_with_linear_SVMs(data_path_1, data_path_2):
    train_X, train_y = load_data(data_path_1)               
    from sklearn.svm import LinearSVC
    classifier = LinearSVC(
        C=10.0,
        tol=0.001,
        verbose=True
    ) 
    classifier.fit(train_X, train_y)

    test_X, test_y = load_data(data_path_2)             
    predict_y = classifier.predict(test_X)
    accuracy = compute_accuracy(predicted_y=predict_y, expected_y=test_y)
    print('Accuracy =', accuracy)

news_groups = Kmeans(20)
news_groups.load_data('../20news-bydate/20news-full-tfidf.txt')
news_groups.run(seed_value="random", criterion="max_iters", threshold=20)           
accuracy = news_groups.compute_purity()
print(accuracy)

clustering_with_KMeans('../20news-bydate/20news-full-tfidf.txt')

classifying_with_linear_SVMs('../20news-bydate/20news-train-tfidf.txt', \
                            '../20news-bydate/20news-test-tfidf.txt')
