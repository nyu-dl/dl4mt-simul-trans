#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Collection of reward functions for Simultaneous Machine Translation
"""
import numpy
from bleu import *


# computing the discounting matrix
gamma = 0.9
maxlen = 100


def compute_discount(gamma, maxlen):
    c = numpy.ones((maxlen,)) * gamma
    c[0] = 1.
    c = c.cumprod()

    C = numpy.triu(numpy.repeat(c[None, :], repeats=maxlen, axis=0))
    C /= c[:, None]
    return C


GAMMA = compute_discount(gamma, maxlen)  # precomputed

def translation_cost(**_k):

    def BLEU():
        q = numpy.zeros((_k['steps'],))
        s = _k['sample']
        r = _k['reference']
        chencherry = SmoothingFunction()
        b = sentence_bleu(r, s, smoothing_function=chencherry.method5)
        q[-1] = b[1]
        return q, b


    return BLEU()




# The general function for rewards (for simultrans):
def return_reward(**_k):

    # ----------------------------------------------------------------- #
    # reward for quality
    # use negative-loglikelihood as the reward (full sentence)
    # we can also use BLEU for quality, but let's try the simplest one'
    #
    @staticmethod
    def _bpe2words(capsw):
        capw = []
        for cc in capsw:
            capw += [cc.replace('@@ ', '')]
        return capw


    def LogLikelihood():
        q = numpy.zeros((_k['steps'],))
        q[-1] = _k['f_cost'](
            _k['ctx0'], _k['x_mask'], _k['y'], _k['y_mask']
        )
        return q


    def NormLogLikelihood():
        q      = LogLikelihood()
        length = _k['y'].shape[0]
        return q / float(length)

    def BLEU():
        q = numpy.zeros((_k['steps'],))
        s = _k['sample']
        r = _k['reference']
        chencherry = SmoothingFunction()
        q[-1] = sentence_bleu(r, s, smoothing_function=chencherry.method5)
        return q

    def BLEUwithForget(beta=None, discount=1., return_quality=False):

        # init
        words = _k['words'].split()  # end-of-sentence is treated as a word
        ref   = _k['reference']

        q0    = numpy.zeros((_k['steps'],))

        # check 0, 1
        maps  = [(it, a) for it, a in enumerate(_k['act']) if a < 2]
        kmap  = len(maps)
        lb    = numpy.zeros((kmap,))
        ts    = numpy.zeros((kmap,))
        q     = numpy.zeros((kmap,))

        if not beta:
            beta = kmap

        beta = 1. / float(beta)

        chencherry = SmoothingFunction()

        # compute BLEU for each Yt
        Y = []
        bleus = []
        truebleus = []

        if len(words) == 0:
            bleus = [0]
            truebleus = [0]

        for t in xrange(len(words)):
            if len(Y) > 0:
                _temp = Y[-1] + ' ' + words[t]
                _temp = _temp.replace('@@ ', '')
                Y = Y[:-1] + _temp.split()
            else:
                Y = [words[t]]

            bb = sentence_bleu(ref, Y, smoothing_function=chencherry.method5)

            bleus.append(bb[1])   # try true BLEU
            truebleus.append(bb[1])


        # print 'Latency BLEU', lbn
        bleus = [0] + bleus    # use TRUE BLEU
        bleus = numpy.array(bleus)
        temp  = bleus[1:] - bleus[:-1]

        tpos  = 0
        for pos, (it, a) in enumerate(maps):
            if (a == 1) and (tpos < len(words)):
                q[pos] = temp[tpos]
                q0[it] = q[pos]
                tpos  += 1

        # add the whole sentence balance on it
        q0[-1] = truebleus[-1]  # the last BLEU we use the real BLEU score.
        return q0



    def LatencyBLEUwithForget(beta=None, discount=1., return_quality=False):

        # init
        words = _k['words'].split()  # end-of-sentence is treated as a word
        ref   = _k['reference']

        q0    = numpy.zeros((_k['steps'],))

        # check 0, 1
        maps  = [(it, a) for it, a in enumerate(_k['act']) if a < 2]
        kmap  = len(maps)
        lb    = numpy.zeros((kmap,))
        ts    = numpy.zeros((kmap,))
        q     = numpy.zeros((kmap,))

        if not beta:
            beta = kmap

        beta = 1. / float(beta)

        chencherry = SmoothingFunction()

        # compute BLEU for each Yt
        Y = []
        bleus = []
        truebleus = []
        for t in xrange(len(words)):
            if len(Y) > 0:
                _temp = Y[-1] + ' ' + words[t]
                _temp = _temp.replace('@@ ', '')
                Y = Y[:-1] + _temp.split()
            else:
                Y = [words[t]]

            bb = sentence_bleu(ref, Y, smoothing_function=chencherry.method5)

            bleus.append(bb[0])
            truebleus.append(bb[1])

        bleus.reverse()
        truebleus.reverse()

        # compute the Latency-Bleu
        T = 0
        Prev = 0
        for i, (it, a) in enumerate(maps):
            # print 'Prev', Prev
            if a == 0:  # WAIT
                T += 1
                if i == 0:
                    lb[i] = 0
                else:
                    lb[i] = lb[i - 1] + Prev
            elif a == 1:
                if i < kmap - 1:
                    lb[i] = lb[i - 1] - Prev

                    Prev = bleus.pop()
                    lb[i] += Prev
                else:
                    lb[i] = lb[i - 2]
            else:
                lb[i] = 0

            ts[i] = T

        # average the score
        # print 'Unnormalized BLEU', lb
        lbn = lb / ts

        # print 'Latency BLEU', lbn
        q[1:] = lbn[1:] - lbn[:-1]
        # print 'instant reward', q

        # add the whole sentence balance on it
        q[-1] = Prev  # the last BLEU
        # print 'instant reward', q

        for i, (it, a) in enumerate(maps):
            q0[it] = q[i]

        return q0


    # ----------------------------------------------------------------- #
    # reward for delay
    # several options:
    # 1. the total delay, which is computed at the last step
    def NormalizedDelay():
        d = numpy.zeros((_k['steps'],))
        # print a
        _src = 0
        _trg = 0
        _sum = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                if _src < _k['source_len']:
                    _src += 1
            elif a == 1:
                _trg += 1
                _sum += _src
        d[-1] = _sum / (_src * _trg + 1e-6)
        return d

    def NormalizedDelay2():
        d = numpy.zeros((_k['steps'],))
        # print a
        _src = 0
        _trg = 0
        _sum = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                if _src < _k['source_len']:
                    _src += 1

            elif a == 1:
                _trg += 1
                _sum += _src
        d[-1] = _sum / ( _k['source_len'] * _trg + 1e-6)
        return d


    # do not use this
    def NormalizedDelaywithPenalty():
        d = numpy.zeros((_k['steps'],))
        a = numpy.array(_k['act'], dtype='float32')
        # print a
        d[-1] = numpy.sum(numpy.cumsum(1 - a) * a) / (_k['src_max'] * numpy.sum(a)) * numpy.exp(-3. / _k['src_max'])
        return d

    def ConsectiveWaiting():
        d = numpy.zeros((_k['steps'],))
        a = numpy.array(_k['act'], dtype='float32')


    def StepDeley():
        d = numpy.array(_k['act'], dtype='float32') - 1.
        return d


    def SilceDelay(win=5):
        d0 = numpy.array(_k['act'], dtype='float32') - 1.

        def slice(m):
            d     = d0
            d[m:] = d0[:-m]
            return  d

        dd = numpy.mean([d0] + [slice(w) for w in range(1, win)])
        return dd

    # -reward of delay
    def MovingDelay(beta=0.1):
        d    = numpy.zeros((_k['steps'],))
        _max = 0
        _cur = 0

        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
                if _cur > _max:
                    _max += 1
                    d[it] = -1
            else:
                _cur = 0

        return d * beta


    def MaximumDelay(_max=5, beta=0.1):
        d    = numpy.zeros((_k['steps'],))
        _cur = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
                if _cur > _max:
                    d[it] = -1
                pass
            elif a == 1:   # only for new commit
                _cur = 0

        return d * beta


    def MaximumDelay2(_max=5, beta=0.1):
        d    = numpy.zeros((_k['steps'],))
        _cur = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
                if _cur > _max:
                    d[it] = -0.1 * (_cur - _max)
                pass
            elif a == 1:   # only for new commit
                _cur = 0

        return d * beta



    def MaximumSource(_max=7, beta=0.1):
        s = numpy.zeros((_k['steps'], ))
        _cur = 0
        _end = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
            elif a == 2:
                _end += 1

            if (_cur - _end) > _max:
                s[it] = -1
        return s * beta

    def MovingSource(beta=0.1):
        s    = numpy.zeros((_k['steps'],))
        _max = 0
        _cur = 0
        _end = 0

        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
            elif a == 2:
                _end += 1

            temp = _cur - _end
            if temp > _max:
                s[it] = -1
                _max = temp

        return s * beta

    def AwardForget(_max=5, beta=0.1):
        s = numpy.zeros((_k['steps'],))
        _cur = 0
        _end = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
            elif a == 2:
                _end += 1

            if ((_cur - _end) >= _max) and (a == 2):
                s[it] = 1
        return s * beta

    def AwardForgetBi(_max=10, _min=4, beta=0.1):
        s = numpy.zeros((_k['steps'],))
        _cur = 0
        _end = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
            elif a == 2:
                _end += 1

            if ((_cur - _end) >= _max) and (a == 2):
                s[it] = 1

            if ((_cur - _end) <= _min) and (a == 2):
                s[it] = -1
        return s * beta

    def AwardForget2(_max=5, beta=0.001):
        s = numpy.zeros((_k['steps'],))
        _cur = 0
        _end = 0
        for it, a in enumerate(_k['act']):
            if a == 0:
                _cur += 1
            elif a == 2:
                _end += 1

            if a == 2:
                s[it] = (_cur - _end - _max) * 2
        return s * beta



    # ----------------------------------------------------------------- #
    # reward for quality + delay
    def Q2D1(alpha=0.5):
        # q = LogLikelihood()
        q = NormLogLikelihood()
        d = NormalizedDelay()

        r = (q ** alpha) * ((1 - d) ** (1 - alpha))
        R = r[::-1].cumsum()[::-1]
        return R, q[-1], d[-1], r[-1]

    def Q2D2(alpha=0.5):
        # q = LogLikelihood()
        q = BLEU()
        d = NormalizedDelaywithPenalty()

        r = (q * alpha) + ((1 - d) * (1 - alpha))
        R = r[::-1].cumsum()[::-1]
        return R, q[-1], d[-1], r[-1]

    def Q2D3(alpha=0.5):
        # q = LogLikelihood()
        q = BLEU()
        d = NormalizedDelay()

        r = q # (q * alpha) + ((1 - d) * (1 - alpha))
        R = r[::-1].cumsum()[::-1]
        return R, q[-1], d[-1], r[-1]

    def Q2D4(alpha=0.5):
        # q = LogLikelihood()
        q = BLEU()
        d = NormalizedDelay()
        d0 = d[-1]
        d[-1] = numpy.exp(-max(d0 - 0.7, 0))
        r = q * d  # (q * alpha) + ((1 - d) * (1 - alpha))
        R = r[::-1].cumsum()[::-1]
        return R, q[-1], d0, r[-1]


    # ---------------------------------------------------------------- #
    # user defined target delay \tau*
    def QualityDelay(tau = 0.5, gamma=3):
        q = LatencyBLEUex(return_quality=True)
        d = NormalizedDelay()

        # just bleu
        bleu  = q[-1]

        # just delay
        delay = d[-1]

        r = q -  gamma * numpy.maximum(d - tau, 0) ** 2  # instant reward
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def FullQualityDelay(tau = 0.5, gamma=10):
        q  = LatencyBLEUex(return_quality=True)
        d  = NormalizedDelay()
        d1 = SilceDelay()

        # just bleu
        bleu = q[-1]

        # just delay
        delay = d[-1]

        r = q  + d1 - gamma * numpy.maximum(d - tau, 0) ** 2  # instant reward
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    # UPDATE: July 11, 2016: we have several varisions::
    def ReturnA():
        # params
        gamma = _k['gamma']
        beta  = 0.1

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu   = q0[-1]

        # just delay
        delay  = d0[-1]

        # use moving-delay + latency bleu (without final BLEU)
        q      = q0
        q[-1]  = 0.
        d      = MovingDelay(beta=beta)

        r      = q + gamma * d
        R      = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def ReturnB():
        # params
        gamma = _k['gamma']
        beta  = 0.1

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (without final BLEU)
        q     = q0
        q[-1] = 0.
        d     = MaximumDelay(_max=4, beta=beta)

        r = q + gamma * d
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def ReturnC():
        # params
        gamma = _k['gamma']
        beta = 0.1

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (with final BLEU)
        q = q0
        d = MaximumDelay(_max=5, beta=beta)

        r = q + gamma * d
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def ReturnD():
        # params
        gamma = _k['gamma']
        beta  = 0.1

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu   = q0[-1]

        # just delay
        delay  = d0[-1]

        # use moving-delay + latency bleu (with final BLEU)
        q      = q0
        d      = MovingDelay(beta=beta)

        r      = q + gamma * d
        R      = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def ReturnE():
        # params
        gamma = _k['gamma']
        beta  = 0.1
        tau   = _k['target']

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (without final BLEU) + global delay
        q     = q0
        q[-1] = 0.
        d     = MaximumDelay(_max=4, beta=beta)
        d[-1]-= numpy.maximum(delay - tau, 0)

        r = q + gamma * d
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    def ReturnF():
        # params
        gamma = _k['gamma']
        beta  = 0.1
        tau   = _k['target']

        q0 = LatencyBLEUex(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (with final BLEU) + global delay
        q = q0
        d = MaximumDelay(_max=5, beta=beta)
        d[-1] -= numpy.maximum(delay - tau, 0) * gamma

        r = q + d
        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, r

    # ---------------------------------------------------------------- #
    def ReturnG():
        # params
        discount = _k['discount']   ## 0.95 here gamma is the discounting factor
        beta = 0.1

        q0 = LatencyBLEUwithForget(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (with final BLEU)
        q = q0
        d = MaximumDelay(_max=4,  beta=beta)
        s = MaximumSource(_max=7, beta=0.01)

        if discount == 1:
            r = q + d + s
            R = r[::-1].cumsum()[::-1]
        else:
            raise NotImplementedError

        return R, bleu, delay, r

    def ReturnH():
        # params
        discount = _k['discount']   ## 0.95 here gamma is the discounting factor
        beta = 0.1

        q0 = LatencyBLEUwithForget(return_quality=True)
        d0 = NormalizedDelay()

        # just bleu
        bleu = q0[-1]

        # just delay
        delay = d0[-1]

        # use maximum-delay + latency bleu (with final BLEU)
        q = q0
        d = MaximumDelay(_max=4,  beta=beta)
        s = MovingSource(beta=0.02)

        if discount == 1:
            r = q + d + s
            R = r[::-1].cumsum()[::-1]
        else:
            raise NotImplementedError

        return R, bleu, delay, r

    def ReturnI():
        # params

        discount = _k['gamma']  ## 0.95 here gamma is the discounting factor
        maxsrc   = _k['maxsrc']
        beta = 0.1

        q0 = LatencyBLEUwithForget(return_quality=True)
        d0 = NormalizedDelay()

        # global reward signal :::>>>
        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # local reward signal :::>>>>
        # use maximum-delay + latency bleu (with final BLEU)
        q     = q0
        q[-1] = 0
        d     = MaximumDelay(_max=5, beta=beta)
        s     = AwardForget(_max=maxsrc, beta=0.01)
        # s     = AwardForget2(_max=maxsrc, beta=0.001)

        r0    = q + d + s
        rg    = bleu      # it is a global reward, will not be discounted.

        if discount == 1:
            r      = r0
            r[-1] += rg
            R      = r[::-1].cumsum()[::-1]
        else:
            R      = numpy.zeros_like(r0)
            R[-1]  = r0[-1]
            for it in range(_k['steps'] - 2, -1, -1):
                R[it] = discount * R[it + 1] + r0[it]
            R      += rg  # add a global signal (without a discount factor)

        return R, bleu, delay, r0

    def ReturnJ():
        # params

        discount = _k['gamma']  ## 0.95 here gamma is the discounting factor
        beta = 0.1

        q0 = LatencyBLEUwithForget(return_quality=True)
        d0 = NormalizedDelay()

        # global reward signal :::>>>
        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # local reward signal :::>>>>
        # use maximum-delay + latency bleu (with final BLEU)
        q     = q0
        q[-1] = 0
        d     = MaximumDelay(_max=5, beta=beta)
        # s     = AwardForget(_max=5, beta=0.01)

        r0    = q + d
        rg    = bleu      # it is a global reward, will not be discounted.

        if discount == 1:
            r      = r0
            r[-1] += rg
            R      = r[::-1].cumsum()[::-1]
        else:
            R      = numpy.zeros_like(r0)
            R[-1]  = r0[-1]
            for it in range(_k['steps'] - 2, -1, -1):
                R[it] = discount * R[it + 1] + r0[it]
            R      += rg  # add a global signal (without a discount factor)

        return R, bleu, delay, r0


    # **------------------------------------------------ **#
    # Finalized Reward function:                           #
    # **------------------------------------------------ **#
    def NewReward():
        # params

        maxsrc   = _k['maxsrc']
        target   = _k['target']
        cw       = _k['cw']
        beta     = 0.03 # 0.5

        q0 = BLEUwithForget(return_quality=True)
        d0 = NormalizedDelay()

        # global reward signal :::>>>
        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # local reward signal :::>>>>
        # use maximum-delay + latency bleu (with final BLEU)
        q = q0
        q[-1] = 0
        if cw > 0:
            d = MaximumDelay2(_max=cw, beta=beta)
        else:
            d = 0

        # s = AwardForget(_max=maxsrc, beta=0.01)
        # s = AwardForgetBi(_max=maxsrc, beta=0.01)

        r0  = q + 0.5 * d

        if target < 1:
            tar = -numpy.maximum(delay - target, 0)
        else:
            tar = 0

        rg  = bleu + tar # it is a global reward, will not be discounted.
        r      = r0
        r[-1] += rg

        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, R[0]


    type  = _k['Rtype']

    funcs = [ReturnA, ReturnB, ReturnC, ReturnD, ReturnE, ReturnF, ReturnG, ReturnH, ReturnI, ReturnJ, NewReward]
    return funcs[type]()

