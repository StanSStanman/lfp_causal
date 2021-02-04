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
