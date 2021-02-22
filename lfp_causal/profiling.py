from threading import Timer
import psutil

class RepeatedTimer(object):
    def __init__(self, interval, function, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        self.function(*self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False


def memory_usage():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    mem_stats['used'].append(vm[3])
    mem_stats['free'].append(vm[4])
    mem_stats['perc'].append(vm[2])
    mem_stats['swap'].append(sw[1])
    return mem_stats


def cpu_usage():
    mcp = psutil.cpu_percent()
    scp = psutil.cpu_percent(percpu=True)
    cpu_stats['mean'].append(mcp)
    for i, c in enumerate(scp):
        cpu_stats['CPU{0}'.format(i)].append(c)
    return cpu_stats


mem_stats = {'used': [],
             'free': [],
             'perc': [],
             'swap': []}


cpu_stats = {'mean': []}
for c in range(len(psutil.cpu_percent(percpu=True))):
    cpu_stats['CPU{0}'.format(c)] = []


# def memory_usage(mem_events=[]):
#     vm = psutil.virtual_memory()
#     sw = psutil.swap_memory()
#     if not isinstance(mem_events, dict):
#         stats = {'used': [],
#                  'free': [],
#                  'perc': [],
#                  'swap': []}
#         mem_events = stats
#     mem_events['used'].append(vm[3])
#     mem_events['free'].append(vm[4])
#     mem_events['perc'].append(vm[4])
#     mem_events['swap'].append(sw[1])
#     return mem_events


# def cpu_usage(cpu_events=[]):
#     mcp = psutil.cpu_percent()
#     scp = psutil.cpu_percent(percpu=True)
#     if not isinstance(cpu_events, dict):
#         stats = {'mean': []}
#         for c in range(len(scp)):
#             stats['CPU{0}'.format(c)] = []
#         cpu_events = stats
#     cpu_events['mean'].append(mcp)
#     for i, c in enumerate(scp):
#         cpu_events['CPU{0}'.format(i)].append(c)
#     return cpu_events


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    import os
    import os.path as op
    import json

    mu = RepeatedTimer(0.1, memory_usage)
    cu = RepeatedTimer(0.1, cpu_usage)

    for i in np.arange(0, 501, 10):
        arr = np.random.uniform(-10., 10., (i, i, i))
        arr = np.divide(arr ** 2, arr, out=arr)
        # time.sleep(1)

    mu.stop()
    cu.stop()

    md = memory_usage().copy()
    cd = cpu_usage().copy()

    t = np.array(range(len(md['perc']))) / (1 / 0.1)

    mfig, (max1, max2) = plt.subplots(1, 2)
    ml1, = max1.plot(t, np.array(md['used']) / (1024 ** 2), label='used RAM (MB)')
    ml1, = max1.plot(t, np.array(md['free']) / (1024 ** 2), label='free RAM (MB)')
    max1.legend()
    ml2, = max2.plot(t, np.array(md['perc']), label='percentage of used RAM')
    max2.legend()
    plt.show()

    cfig, (cax1, cax2) = plt.subplots(1, 2)
    cl1, = cax1.plot(t, np.array(cd['mean']), label='mean CPU usage')
    cax1.legend()
    for k in cd.keys():
        if k.startswith('CPU'):
            cl2, = cax2.plot(t, np.array(cd[k]), label=k)
    cax2.legend()
    plt.show()


    print('ciao')
    # ftm = time.strftime('%d%m%y%H%M%S', time.localtime())
    # running = 'fts0_fr_ea'
    # m_out_dir = op.join('/home', 'rbasanisi', 'profiling', 'memory', running)
    # c_out_dir = op.join('/home', 'rbasanisi', 'profiling', 'cpu', running)
    # for d in [m_out_dir, c_out_dir]:
    #     os.makedirs(d, exist_ok=True)
    # m_out_file = op.join(m_out_dir, 'memory_test_{0}.json'.format(ftm))
    # c_out_file = op.join(c_out_dir, 'cpu_test_{0}.json'.format(ftm))
    # for jfn, td in zip([m_out_file, c_out_file],
    #                    [memory_usage(), cpu_usage()]):
    #     with open(jfn, 'w') as jf:
    #         json.dump(td, jf)
