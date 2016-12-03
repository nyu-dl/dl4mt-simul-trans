# -*- coding: utf-8 -*-
import matplotlib
# matplotlib.use('agg')
import copy
import numpy
import os
import seaborn as sns
import pandas as pd
sns.set(context="paper", font="monospace", style='whitegrid')
from matplotlib import pyplot as plot
from matplotlib import rc

rc('font',**{'family':'Verdana', 'weight': 'normal'})
rc('font', size=8)
rc('text', usetex=True)
rc('text.latex',unicode=True)
rc('text.latex',preamble='\usepackage[utf8]{inputenc}')
rc('text.latex',preamble='\usepackage[russian]{babel}')
rc('text.latex',preamble='\usepackage[german]{babel}')
rc('text.latex',preamble='\usepackage[ngerman]{babel}')

matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['xtick.labelsize'] = 11 

def heatmap(sources, refs, trans, actions, idx, atten=None, savefig=True, name='test', info=None, show=False):
    source = [s.strip() for s in sources[idx].decode('utf8').replace('@@', '--').split()] + ['||']
    target = ['*'] + [s.strip() for s in trans[idx].decode('utf8').replace('@@', '--').split()] + ['||']
    action = actions[idx]
   

    if atten:
        attention = numpy.array(atten[idx])

    def track(acts, data, annote):
        x, y = 0, 0
        for a in acts:
            x += a
            y += 1 - a
            # print a, x, y, target[x].encode('utf8')
            data[y, x]   = 1
            annote[y, x] = 'W' if a == 0  else 'C'

        return data, annote
    # print target
    
    data       = numpy.zeros((len(source), len(target)))
    annote     = numpy.chararray(data.shape, itemsize=8)
    annote[:]  = '' 
    data, annote  = track(action, data, annote)
    data[0, 0]    = 1
    annote[0, 0]  = 'S'
    if atten:
        data[:-1, 1:] += attention.T
    
    d  = pd.DataFrame(data=data, columns=target, index=source)
    # p  = sns.diverging_palette(220, 10, as_cmap=True)
    f, ax = plot.subplots(figsize=(11, 11))
    f.set_canvas(plot.gcf().canvas)
    g  = sns.heatmap(d, ax=ax, annot=annote, fmt='s')
    g.xaxis.tick_top()

    plot.xticks(rotation=90)
    plot.yticks(rotation=0)
    # plot.show()
    if savefig:
        if not os.path.exists('.images/C_{}'.format(name)):
            os.mkdir('.images/C_{}'.format(name))

        filename = 'Idx={}||'.format(info['index'])
        for w in info:
            if w is not 'index':
                filename += '.{}={:.2f}'.format(w, float(info[w]))

        print 'saving...'
        f.savefig('.images/C_{}'.format(name) + '/{}'.format(filename) + '.pdf', dpi=100)
    if show:
        plot.show()

    print 'plotting done.'
    plot.close()
 
def heatmap2(sources, refs, trans, actions, idx, atten=None, full_atten=None, savefig=True, name='test', info=None, show=False):
    source = ['*'] + [s.strip() for s in sources[idx].decode('utf8').replace('@@', '--').split()] + ['||']
    target = ['*'] + [s.strip() for s in trans[idx].decode('utf8').replace('@@', '--').split()] + ['||'] + ['*']
    action = actions[idx]
   
    flag   = 0
    if atten:
        attention = numpy.array(atten[idx])
    else:
        attention = None

    if full_atten:
        fullatten = numpy.array(full_atten[idx])
    else:
        fullatten = None
    
    def track(acts, data, annote):
        x, y, z = 0, 0, 0
        for a in acts:
            x += (a == 1)
            y += (a == 0)
            z += (a == 2)

            # data[y + 1, x]   = 1
            # data[z, x + 1]   = 1
            # annote[y, x] = 'W' if a == 0  else 'C'

        return data, annote
    # print target
    
    data       = numpy.zeros((len(source), len(target)))
    annote     = numpy.chararray(data.shape, itemsize=8)
    annote[:]  = '' 
    data, annote  = track(action, data, annote)
    data[1, 0] = 1
    
    def draw(data_t, ax, attention=None):
        
        data   = copy.copy(data_t)
        data[1:-1, 1:-1] += attention.T
        d  = pd.DataFrame(data=data, columns=target, index=source)
        # p  = sns.diverging_palette(220, 10, as_cmap=True)
        g  = sns.heatmap(d, mask=(data==0), square=True, cbar=False, linewidths=0.1, ax=ax, annot=annote, fmt='s')
        g.xaxis.tick_top()
   
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
        
        ax.grid(True)
    f, [ax1, ax2] = plot.subplots(1, 2, figsize=(22, 11))
    f.set_canvas(plot.gcf().canvas)
    
    draw(data, ax1, attention)
    # plot.xticks(rotation=90)
    # plot.yticks(rotation=0)
    # plot.grid()
    
    draw(data, ax2, fullatten)
    # plot.xticks(rotation=90)
    # plot.yticks(rotation=0)
    # plot.grid()

    
    if savefig:
        if not os.path.exists('.images/M_{}'.format(name)):
            os.mkdir('.images/M_{}'.format(name))

        filename = 'Idx={}||'.format(info['index'])
        for w in info:
            if w is not 'index':
                filename += '.{}={:.2f}'.format(w, float(info[w]))

        # print 'saving...'
        plot.savefig('.images/M_{}'.format(name) + '/{}'.format(filename) + '.pdf', dpi=100)
    
    if show:
        plot.show()

    # print 'plotting done.'
    plot.close()
 





def visualize(sources, refs, trans, aligns, idx, savefig=True, name='test', info=None):
    
    colors = ['b', 'g']

    fig = plot.figure(figsize=(20, 2))
    ax = plot.gca()

    # plot.hold('on')

    plot.xlim([0., 10.])

    scolors = []
    caidx = 0
    coloridx = 0
    for sidx in xrange(len([s_.replace('@@', '--').strip() for s_ in sources[idx].split()] + ['<eos>'])):
        if caidx >= len(numpy.unique(aligns[idx])) or sidx >= numpy.unique(aligns[idx])[caidx]:
            caidx = caidx + 1
            coloridx = 1 - coloridx
        scolors.append(colors[coloridx])

    tcolors = []
    lastidx = -1
    coloridx = 1
    for tt in aligns[idx]:
        if tt != lastidx:
            lastidx = tt
            coloridx = 1 - coloridx
        tcolors.append(colors[coloridx])

    x, y = 0., 1.
    s_pos = [(x, y)]
    for ii, ss in enumerate([s_.replace('@@', '--').strip() for s_ in sources[idx].split()] + ['<eos>']):
        
        ss.replace('%', '\%')
        xx = plot.text(x, y, ss)
        xx.set_bbox(dict(color=scolors[ii], alpha=0.1, edgecolor=scolors[ii]))
        xx._renderer = fig.canvas.get_renderer()
        wext = xx.get_window_extent()
        bbox = ax.transData.inverted().transform(wext)
        x = bbox[1, 0] + 0.
        s_pos.append((x, y))
    s_pos.append((bbox[1, 0], y))

    x, y = 0., .95
    t_pos = []
    for ii, ss in enumerate([s_.decode('utf8').replace('@@', '--') for s_ in trans[idx].split()]):
        
        ss.replace('%', '\%')
        xx = plot.text(x, y, ss)
        xx._renderer = fig.canvas.get_renderer()
        wext = xx.get_window_extent()
        bbox = ax.transData.inverted().transform(wext)
        t_pos.append((bbox[0, 0], bbox[0, 1] + 0.03))
        x = bbox[1, 0] + 0.
    t_pos.append((bbox[1, 0], bbox[0, 1] + 0.03))

    lasttidx = 0
    lastidx = -1
    for tidx, sidx in enumerate(aligns[idx]):
        if lastidx != sidx:
            lastidx = sidx
            lasttidx = tidx
            sidx = numpy.minimum(sidx, len(s_pos) - 1)
            plot.arrow(s_pos[sidx][0], s_pos[sidx][1],
                       t_pos[tidx][0] - s_pos[sidx][0],
                       t_pos[tidx][1] - s_pos[sidx][1],
                       head_width=0., head_length=0.,
                       fc=tcolors[tidx], ec=tcolors[tidx],
                       linestyle='dotted', width=0.0001)
            for tt in xrange(tidx, len(aligns[idx])):
                if aligns[idx][tt] != sidx:
                    plot.arrow(s_pos[sidx][0], s_pos[sidx][1],
                               t_pos[tt][0] - s_pos[sidx][0],
                               t_pos[tt][1] - s_pos[sidx][1],
                               head_width=0., head_length=0.,
                               fc=tcolors[tidx], ec=tcolors[tidx],
                               linestyle='dotted', width=0.0001)
                    plot.fill_between([t_pos[tidx][0], s_pos[sidx][0], t_pos[tt][0]],
                                      [t_pos[tidx][1], s_pos[sidx][1], t_pos[tt][1]],
                                      facecolor=tcolors[tidx], alpha=0.1)
                    break
    plot.arrow(s_pos[sidx][0], s_pos[sidx][1],
               t_pos[-1][0] - s_pos[sidx][0],
               t_pos[-1][1] - s_pos[sidx][1],
               head_width=0., head_length=0.,
               fc=tcolors[-1], ec=tcolors[-1],
               linestyle='dotted', width=0.0001)
    plot.fill_between([t_pos[lasttidx][0], s_pos[sidx][0], t_pos[-1][0]],
                      [t_pos[lasttidx][1], s_pos[sidx][1], t_pos[-1][1]],
                      facecolor=tcolors[tidx], alpha=0.1)

    # plot.hold('off')

    plot.axis('off')
    plot.ylim([0.95, 1.01])
    plot.tight_layout()

    if savefig:
        if not os.path.exists('.images/{}'.format(name)):
            os.mkdir('.images/{}'.format(name))

        filename = 'Idx={}||'.format(info['index'])
        for w in info:
            if w is not 'index':
                filename += '.{}={:.2f}'.format(w, float(info[w]))

    plot.savefig('.images/{}'.format(name) + '/{}'.format(filename) + '.pdf', dpi=300)

    print 'plotting done.'
    plot.close()
    # plot.show()


if __name__ == "__main__":

    sources = ['I cannot understand .']
    targets = ['Ich verstehe nicht .']
    actions = [[0, 0, 1, 1, 2, 0, 1, 2, 2,  0, 1]]
    heatmap2(sources, targets, targets, actions, 0, savefig=False, show=True)
