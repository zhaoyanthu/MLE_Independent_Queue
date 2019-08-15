import numpy as np

class QLsimulator(object):
    """ A simulator to generate queues and partial queues
    """
    def __init__(self, mode, iteration, p, r, seed=0, isOverflow=False, service_rate=10000, rg=None):
        """ Constructor of the Queue Length Simulator
        Args:
            mode: string, mode of stochastic process, options are within {'poisson', 'uniform', 'normal'}
            iteration: int, number of iterations/cycles
            p: float, penetration rate
            r: int, lambda of poisson distribution, arrival rate in the red phase
            isOverflow: bool, indicate if it is oversaturation
            service_rate: int, average service rate of the queue (now it is a const, however can be random)
            rg: arrival rate in the green phase (only used in over-saturated condition)
        Returns:
            None
        """
        self.mode = mode
        self.iteration = iteration
        self.p = p
        self.r = r
        self.seed = seed
        self.isOverflow = isOverflow
        self.service_rate = service_rate
        self.rg = rg if rg != None else self.r
        
        if self.mode not in ['poisson', 'uniform', 'normal']:
            print('Inproper stochastic process!!')
            raise(ValueError)
        
        
    def __initialize_data(self):
        """ Initialize the data needed for the simulator
        Args:
            None
        Returns:
            None
        """
        # calculated in function generate_queues()
        self.real_queue = []
        self.obs_queue = []
        self.pos_all_dist = {}
        self.pos_cv_dist = {}
        self.firsts = []
        self.lasts = [] # this is not the same with self.obs_queue_length because it may contains None
        
        # calculated in function generate_queues()
        self.real_queue_length = []
        self.obs_queue_length = []
        self.obs_cvs = []
        self.total_all = 0
        self.total_cv = 0
        self.real_queue_stat = {}
        self.obs_queue_stat = {}
        self.num_empty = 0
        self.L_max = 0

    
    def __generate_a_queue_oversaturation(self, last_N=0, last_veh_type=np.array([])):
        """ Generate a new queue, given the residual queue
        Args:
            last_N, int, residual queue length
            last_veh_type, np.array(int), details of the residual queue
        Returns:
            N, int, queue length
            veh_type, np.array(int), details of the queue (1-CV, 0-regular)
            res_N, int, residual queue length
            res_veh_type, np.array(int), details of the residual queue
        """
        if self.mode=='poisson':
            Ng = np.random.poisson(self.rg)
            Nr = np.random.poisson(self.r)
        elif self.mode=='uniform':
            Ng = np.random.randint(self.rg)
            Nr = np.random.randint(self.r)
        elif self.mode=='normal':
            Ng = int(np.random.normal(self.rg) % (2 * self.rg))
            Nr = int(np.random.normal(self.r) % (2 * self.r))

        if Ng + last_N <= self.service_rate:
            N = Nr
            veh_type = np.random.binomial(1, self.p, Nr)
        else:
            res_N = (Ng + last_N) - self.service_rate
            veh_type_green = np.random.binomial(1, self.p, Ng)
            res_veh_type = np.append(last_veh_type, veh_type_green)[self.service_rate:]
            N = res_N + Nr
            veh_type = np.append(res_veh_type, np.random.binomial(1, self.p, Nr))
            #print(res_N, res_veh_type)
        #print(Ng, Nr, N, veh_type)
        return N, veh_type


    def __generate_a_queue_undersaturation(self):
        """ Generate a new queue, given the residual queue
        Args:
            None
        Returns:
            N, int, queue length
            veh_type, np.array(int), details of the queue (1-CV, 0-regular)
        """
        if self.mode=='poisson':
            N = np.random.poisson(self.r)
        elif self.mode=='uniform':
            N = np.random.randint(self.r)
        elif self.mode=='normal':
            N = int(np.random.normal(self.r) % (2 * self.r))
        veh_type = np.random.binomial(1, self.p, N)
        return N, veh_type

    
    def generate_queues(self, stats=True, data=False):
        """ Generate queues
        Args:
            data: bool, if return or not
        Returns:
            None
        """
        np.random.seed(self.seed)
        self.__initialize_data()
        # Initialize a queue if overflow
        if self.isOverflow:
            N, veh_type = self.__generate_a_queue_oversaturation(0, np.array([]))

        for i in range(self.iteration):
            if self.isOverflow:
                N, veh_type = self.__generate_a_queue_oversaturation(N, veh_type)
            else:
                N, veh_type = self.__generate_a_queue_undersaturation()

            self.real_queue.append(veh_type)
            cv_pos = np.where(veh_type > 0)[0]
            
            if not len(cv_pos):
                self.firsts.append(None)
                self.lasts.append(None)
                self.obs_queue.append([])
            else: 
                first, last = cv_pos[0] + 1, cv_pos[-1] + 1
                self.firsts.append(first)
                self.lasts.append(last)
                self.obs_queue.append(veh_type[:last])
                
            # Get the distributions of all the stopped vehicles
            for pos in range(N):
                self.pos_all_dist[pos + 1] = self.pos_all_dist.get(pos + 1, 0) + 1
            for pos in cv_pos:
                self.pos_cv_dist[pos + 1] = self.pos_cv_dist.get(pos + 1, 0) + 1

        if stats:
            self.__cal_stats(False)
        if data:
            return self.real_queue, self.obs_queue, self.pos_all_dist, self.pos_cv_dist
        
        
    def __cal_stats(self, data=False):
        """ Calculate some statistics of the queues
        Args:
            data: bool, if return or not
        Returns:
            self.real_queue_length: [int], collection of queue lengths
            self.obs_queue_length: [int], collection of observed queue lengths
            self.real_queue_length: {length:count}, counts of queue lengths
            self.obs_queue_length: {length:count}, counts of observed queue lengths
            self.total_all: int, total number of vehicles
            self.total_cv: int, total number of CVs
        """
        assert([sum(q) for q in self.obs_queue] == [sum(q) for q in self.real_queue])
        for q in self.real_queue:
            self.real_queue_length.append(len(q))
            self.real_queue_stat[len(q)] = self.real_queue_stat.get(len(q), 0) + 1
        for q in self.obs_queue:
            self.obs_queue_length.append(len(q))
            self.obs_cvs.append(int(sum(q)))
            if not len(q): 
                self.num_empty += 1
            self.obs_queue_stat[len(q)] = self.obs_queue_stat.get(len(q), 0) + 1
        self.total_all = sum(self.real_queue_length)
        self.total_cv = sum(self.obs_cvs)
        self.L_max = max(self.obs_queue_length, default=0) 
        if data:
            return (self.real_queue_length, self.obs_queue_length, 
                    self.real_queue_stat, self.obs_queue_stat,
                    self.total_all, self.total_cv)

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Test overflow queues
    for service_rate in [22]:
        simulator = QLsimulator(mode='poisson', iteration=10, p=0.1, r=10, isOverflow=True, service_rate=service_rate)
        simulator.generate_queues()
        plt.hist(simulator.real_queue_length, bins=range(40), alpha=0.2)
        plt.title('Service rate: ' + str(simulator.service_rate))
        plt.show()