"""
-- Policy Network for decision making [more general]
"""
from nmt_uni import *
from layers import _p

import os
import time, datetime
import cPickle as pkl

# hyper params
TINY = 1e-7
PI   = numpy.pi
E    = numpy.e
A    = 0.2
B    = 1


class Controller(object):

    def __init__(self, trng,
                 options,
                 n_in=None, n_out=None,
                 recurrent=False, id=None):

        self.WORK      = options['workspace']

        self.trng      = trng
        self.options   = options
        self.recurrent = recurrent
        self.type      = options.get('type', 'categorical')

        self.n_hidden  = 128
        self.n_in      = n_in
        self.n_out     = n_out

        if self.options.get('layernorm', True):
            self.rec = 'lngru'
        else:
            self.rec = 'gru'

        if not n_in:
            self.n_in  = options['readout_dim']

        if not n_out:
            if self.type == 'categorical':
                self.n_out = 2    # initially it is a WAIT/COMMIT action.
            elif self.type == 'gaussian':
                self.n_out = 100
            else:
                raise NotImplementedError

        # build the policy network
        print 'parameter initialization'

        params = OrderedDict()

        if not self.recurrent:
            print 'building a feedforward controller'
            params = get_layer('ff')[0](options, params, prefix='policy_net_in',
                                        nin=self.n_in, nout=self.n_hidden)
        else:
            print 'building a recurrent controller'
            params = get_layer(self.rec)[0](options, params, prefix='policy_net_in',
                                         nin=self.n_in, dim=self.n_hidden)

        params = get_layer('ff')[0](options, params, prefix='policy_net_out',
                                    nin=self.n_hidden,
                                    nout=self.n_out if self.type == 'categorical' else self.n_out * 2)

        # bias the forget probability
        # if self.n_out == 3:
        #    params[_p('policy_net_out', 'b')][-1]  = -2

        # for the baseline network.
        params_b = OrderedDict()

        # using a scalar baseline [**]
        # params_b['b0'] = numpy.array(numpy.random.rand() * 0.0, dtype='float32')

        # using a MLP as a baseline
        params_b = get_layer('ff')[0](options, params_b, prefix='baseline_net_in',
                                      nin=self.n_in, nout=128)
        params_b = get_layer('ff')[0](options, params_b, prefix='baseline_net_out',
                                      nin=128, nout=1)

        if id is not None:
            print 'reload the saved model: {}'.format(id)
            params   = load_params(self.WORK + '.policy/{}-{}.current.npz'.format(id, self.options['base']), params)
            params_b = load_params(self.WORK + '.policy/{}-{}.current.npz'.format(id, self.options['base']), params_b)
        else:
            id = datetime.datetime.fromtimestamp(time.time()).strftime('%y%m%d-%H%M%S')
            print 'start from a new model: {}'.format(id)

        self.id = id
        self.model = self.WORK + '.policy/{}-{}'.format(id, self.options['base'])

        # theano shared params
        tparams        = init_tparams(params)
        tparams_b      = init_tparams(params_b)

        self.tparams   = tparams
        self.tparams_b = tparams_b

        # build the policy network
        self.build_sampler(options=options)
        self.build_discriminator(options=options)

        print 'policy network'
        for p in params:
            print p, params[p].shape

    def build_batchnorm(self, observation, mask=None):
        raise NotImplementedError

    def build_sampler(self, options):

        # ==================================================================================== #
        # Build Action function: samplers
        # ==================================================================================== #

        observation = tensor.matrix('observation', dtype='float32')  # batch_size x readout_dim (seq_steps=1)
        prev_hidden = tensor.matrix('p_hidden', dtype='float32')

        if not self.recurrent:
            hiddens = get_layer('ff')[1](self.tparams, observation,
                                         options, prefix='policy_net_in',
                                         activ='tanh')
        else:
            hiddens = get_layer(self.rec)[1](self.tparams, observation,
                                             options, prefix='policy_net_in', mask=None,
                                             one_step=True, _init_state=prev_hidden)[0]

        act_inps = [observation, prev_hidden]
        if self.type == 'categorical':
            act_prob  = get_layer('ff')[1](self.tparams, hiddens, options,
                                           prefix='policy_net_out',
                                           activ='softmax')   # batch_size x n_out
            act_prob2 = tensor.clip(act_prob, TINY, 1 - TINY)

            # compiling the sampling function for action
            # action        = self.trng.binomial(size=act_prop.shape, p=act_prop)
            action          = self.trng.multinomial(pvals=act_prob).argmax(1)     # 0, 1, ...

            print 'build action sampling function [Discrete]'
            self.f_action = theano.function(act_inps, [action, act_prob, hiddens, act_prob2],
                                            on_unused_input='ignore')  # action/dist/hiddens

        elif self.type == 'gaussian':
            _temp = get_layer('ff')[1](self.tparams, hiddens, options,
                                         prefix='policy_net_out',
                                         activ='linear'
                                        )   # batch_size x n_out
            mean, log_std = _temp[:, :self.n_out], _temp[:, self.n_out:]
            mean, log_std = -A * tanh(mean), -B-relu(log_std)

            action0       = self.trng.normal(size=mean.shape, dtype='float32')
            action        = action0 * tensor.exp(log_std) + mean

            print 'build action sampling function [Gaussian]'
            self.f_action = theano.function(act_inps, [action, mean, log_std, hiddens],
                                            on_unused_input='ignore')  # action/dist/hiddens
        else:
            raise NotImplementedError

    def build_discriminator(self, options):
        # ==================================================================================== #
        # Build Action Discriminator
        # ==================================================================================== #

        observations  = tensor.tensor3('observations', dtype='float32')
        mask          = tensor.matrix('mask', dtype='float32')
        if self.type == 'categorical':
            actions       = tensor.matrix('actions', dtype='int64')
        elif self.type == 'gaussian':
            actions       = tensor.tensor3('actions', dtype='float32')
        else:
            raise NotImplementedError

        if not self.recurrent:
            hiddens   = get_layer('ff')[1](self.tparams, observations,
                                         options, prefix='policy_net_in',
                                         activ='tanh')
        else:
            hiddens   = get_layer(self.rec)[1](self.tparams, observations,
                                            options, prefix='policy_net_in', mask=mask)[0]

        act_inputs = [observations, mask]
        if self.type == 'categorical':
            act_probs     = get_layer('ff')[1](self.tparams, hiddens, options, prefix='policy_net_out',
                                               activ='softmax') # seq_steps x batch_size x n_out

            act_probs = tensor.clip(act_probs, TINY, 1 - TINY)

            print 'build action distribiution'
            self.f_probs  = theano.function(act_inputs, act_probs,
                                           on_unused_input='ignore')  # get the action probabilities
        elif self.type == 'gaussian':
            _temps = get_layer('ff')[1](self.tparams, hiddens, options,
                                       prefix='policy_net_out',
                                       activ='linear'
                                       )  # batch_size x n_out
            means, log_stds = _temps[:, :, :self.n_out], _temps[:, :, self.n_out:]
            means, log_stds = -A * tanh(means), -B-relu(log_stds)

            act_probs     = [means, log_stds]

            print 'build Gaussian PDF'
            self.f_pdf    = theano.function(act_inputs, [means, log_stds],
                                           on_unused_input='ignore')  # get the action probabilities
        else:
            raise NotImplementedError

        # ==================================================================================== #
        # Build Baseline Network (Input-dependent Value Function) & Advantages
        # ==================================================================================== #

        print 'setup the advantages & baseline network'
        reward        = tensor.matrix('reward')  # seq_steps x batch_size :: rewards for each steps

        # baseline is estimated with a 2-layer neural network.
        hiddens_b     = get_layer('ff')[1](self.tparams_b, observations, options,
                                           prefix='baseline_net_in',
                                           activ='tanh')
        baseline      = get_layer('ff')[1](self.tparams_b, hiddens_b, options,
                                           prefix='baseline_net_out',
                                           activ='linear')[:, :, 0]  # seq_steps x batch_size or batch_size
        advantages    = self.build_advantages(act_inputs, reward, baseline, normalize=True)


        # ==================================================================================== #
        # Build Policy Gradient (here we provide two options)
        # ==================================================================================== #
        if self.options['updater'] == 'REINFORCE':
            print 'build RENIFROCE.'
            self.build_reinforce(act_inputs, act_probs, actions, advantages)

        elif self.options['updater'] == 'TRPO':
            print 'build TRPO'
            self.build_trpo(act_inputs, act_probs, actions, advantages)
        else:
            raise NotImplementedError

    # ==================================================================================== #
    # Controller Actions
    # ==================================================================================== #
    def random(self, states, p=0.5):
        live_k = states.shape[0]
        return (numpy.random.random(live_k) > p).astype('int64'), \
               numpy.ones(live_k) * p

    def action(self, states, prevhidden):
        return self.f_action(states, prevhidden)

    def init_hidden(self, n_samples=1):
        return numpy.zeros((n_samples, self.n_hidden), dtype='float32')

    def init_action(self, n_samples=1):
        states0 = numpy.zeros((n_samples, self.n_in), dtype='float32')
        return self.f_action(states0, self.init_hidden(n_samples))

    def get_learner(self):
        if self.options['updater'] == 'REINFORCE':
            return self.run_reinforce
        elif self.options['updater'] == 'TRPO':
            return self.run_trpo
        else:
            raise NotImplementedError

    @staticmethod
    def kl(prob0, prob1):
        p1 = (prob0 + TINY) / (prob1 + TINY)
        # p2 = (1 - prob0 + TINY) / (1 - prob1 + TINY)
        return tensor.sum(prob0 * tensor.log(p1), axis=-1)

    @staticmethod
    def _grab_prob(probs, X):
        assert probs.ndim == 3

        batch_size = probs.shape[1]
        max_len    = probs.shape[0]
        vocab_size = probs.shape[2]

        probs = probs.reshape((batch_size * max_len, vocab_size))
        return probs[tensor.arange(batch_size * max_len), X.flatten(1)].reshape(X.shape)  # advanced indexing

    def cross(self, probs, actions):
        # return tensor.log(probs) * actions + tensor.log(1 - probs) * (1 - actions)
        return self._grab_prob(tensor.log(probs), actions)

    def build_advantages(self, act_inputs, reward, baseline, normalize=True):
        # TODO: maybe we need a discount factor gamma for advantages.
        # TODO: we can also rewrite advantages with value functions (GAE)

        # Advantages and Normalization the return
        reward_adv  = reward - baseline
        mask        = act_inputs[1]

        if normalize:
            reward_mean  = tensor.sum(mask * reward_adv) / tensor.sum(mask)
            reward_mean2 = tensor.sum(mask * (reward_adv ** 2)) / tensor.sum(mask)
            reward_std   = tensor.sqrt(tensor.maximum(reward_mean2 - reward_mean ** 2, TINY)) + TINY
            # reward_std  = tensor.maximum(reward_std, 1)
            reward_c     = reward_adv - reward_mean  # independent mean
            advantages   = reward_c / reward_std
        else:
            advantages   = reward_adv

        print 'build advantages and baseline gradient'
        L      = tensor.sum(mask * (reward_adv ** 2)) / tensor.sum(mask)
        dL     = tensor.grad(L, wrt=itemlist(self.tparams_b))
        lr     = tensor.scalar(name='lr')

        inps_b = act_inputs + [reward]
        oups_b = [L, advantages]
        f_adv, f_update_b = adam(lr, self.tparams_b, dL, inps_b, oups_b)

        self.f_adv      = f_adv
        self.f_update_b = f_update_b

        return advantages

    # ===================================================================
    # Policy Grident: REINFORCE with Adam
    # ===================================================================
    def build_reinforce(self, act_inputs, act_probs, actions, advantages):

        mask          = act_inputs[1]

        if self.type == 'categorical':
            negEntropy  = tensor.sum(tensor.log(act_probs) * act_probs, axis=-1)
            logLikelihood   = self.cross(act_probs, actions)

        elif self.type == 'gaussian':
            means, log_stds = act_probs
            negEntropy      = -tensor.sum(log_stds + tensor.log(tensor.sqrt(2 * PI * E)), axis=-1)

            actions0        = (actions - means) / tensor.exp(log_stds)
            logLikelihood   = -tensor.sum(log_stds, axis=-1) - \
                              0.5 * tensor.sum(tensor.sqr(actions0), axis=-1) - \
                              0.5 * means.shape[-1] * tensor.log(2 * PI)

        else:
            raise NotImplementedError

        # tensor.log(act_probs) * actions + tensor.log(1 - act_probs) * (1 - actions)

        H     = tensor.sum(mask * negEntropy, axis=0).mean() * 0.01  # entropy penalty
        J     = tensor.sum(mask * -logLikelihood * advantages, axis=0).mean() + H
        dJ    = grad_clip(tensor.grad(J, wrt=itemlist(self.tparams)))

        print 'build REINFORCE optimizer'
        lr    = tensor.scalar(name='lr')

        inps  = act_inputs + [actions, advantages]
        outps = [J, H]
        if self.type == 'gaussian':
            outps += [actions0.mean(), actions.mean()]

        f_cost, f_update = adam(lr, self.tparams, dJ, inps, outps)

        self.f_cost   = f_cost
        self.f_update = f_update
        print 'done'

    def run_reinforce(self, act_inputs, actions, reward, update=True, lr=0.0002):

        # sub baseline
        inps_adv      = act_inputs + [reward]
        L, advantages = self.f_adv(*inps_adv)
        inps_reinfoce = act_inputs + [actions, advantages]

        if self.type == 'gaussian':
            J, H, m, s    = self.f_cost(*inps_reinfoce)
            info = {'J': J, 'G_norm': H, 'B_loss': L, 'Adv': advantages.mean(), 'm': m, 's': s}
        else:
            J, H = self.f_cost(*inps_reinfoce)
            info = {'J': J, 'B_loss': L, 'Adv': advantages.mean()}

        info['advantages'] = advantages

        if update:  # update the parameters
            self.f_update_b(lr)
            self.f_update(lr)

        return info

    # ==================================================================================== #
    # Trust Region Policy Optimization
    # ==================================================================================== #
    def build_trpo(self, act_inputs, act_probs, actions, advantages):

        assert self.type == 'categorical', 'in this stage not support TRPO'

        # probability distribution
        mask      = act_inputs[1]
        probs     = act_probs
        probs_old = tensor.matrix(dtype='float32')

        logp      = self.cross(probs, actions)
        logp_old  = self.cross(probs_old, actions)

        # policy gradient
        J         = tensor.sum(mask * -tensor.exp(logp - logp_old) * advantages, axis=0).mean()
        dJ        = flatgrad(J, self.tparams)
        probs_fix = theano.gradient.disconnected_grad(probs)

        kl_fix    = tensor.sum(mask * self.kl(probs_fix, probs), axis=0).mean()
        kl_grads  = tensor.grad(kl_fix, wrt=itemlist(self.tparams))
        ftangents = tensor.fvector(name='flat_tan')
        shapes    = [self.tparams[var].get_value(borrow=True).shape for var in self.tparams]
        start     = 0
        tangents  = []
        for shape in shapes:
            size = numpy.prod(shape)
            tangents.append(tensor.reshape(ftangents[start:start + size], shape))
            start += size
        gvp       = tensor.add(*[tensor.sum(g * t) for (g, t) in zipsame(kl_grads, tangents)])

        # Fisher-vectror product
        fvp       = flatgrad(gvp, self.tparams)
        entropy   = tensor.sum(mask * -self.cross(probs, probs), axis=0).mean()
        kl        = tensor.sum(mask * self.kl(probs_old, probs), axis=0).mean()

        print 'compile the functions'
        inps          = act_inputs + [actions, advantages, probs_old]
        loss          = [J, kl, entropy]
        self.f_pg     = theano.function(inps, dJ)
        self.f_loss   = theano.function(inps, loss)
        self.f_fisher = theano.function([ftangents] + inps, fvp, on_unused_input='ignore')

        # get/set flatten params
        print 'compling flat updater'
        self.get_flat = theano.function([], tensor.concatenate([self.tparams[v].flatten() for v in self.tparams]))
        theta         = tensor.vector()
        start         = 0
        updates       = []
        for v in self.tparams:
            p     = self.tparams[v]
            shape = p.shape
            size  = tensor.prod(shape)
            updates.append((p, theta[start:start + size].reshape(shape)))
            start += size
        self.set_flat = theano.function([theta], [], updates=updates)


    def run_trpo(self, act_inputs, actions, reward,
                 update=True, cg_damping=1e-3, max_kl=1e-2, lr=0.0002):

        # sub baseline
        inps_adv      = act_inputs + [reward]
        L, advantages = self.f_adv(*inps_adv)
        self.f_update_b(lr)

        # get current action distributions
        probs  = self.f_probs(*act_inputs)
        inps   = act_inputs + [actions, advantages, probs]
        thprev = self.get_flat()

        def fisher_vector_product(p):
            return self.f_fisher(p, *inps) + cg_damping * p

        g             = self.f_pg(*inps)
        losses_before = self.f_loss(*inps)

        if numpy.allclose(g, 0):
            print 'zero gradient, not updating'
        else:
            stepdir  = self.cg(fisher_vector_product, -g)
            shs      = .5 * stepdir.dot(fisher_vector_product(stepdir))
            lm       = numpy.sqrt(shs / max_kl)

            print "\nlagrange multiplier:", lm, "gnorm:", numpy.linalg.norm(g)
            fullstep       = stepdir / lm
            neggdotstepdir = -g.dot(stepdir)

            def loss(th):
                self.set_flat(th)
                return self.f_loss(*inps)[0]

            print 'do line search'
            success, theta = self.linesearch(loss, thprev, fullstep, neggdotstepdir / lm)

            print "success", success
            self.set_flat(theta)

        losses_after = self.f_loss(*inps)

        info = OrderedDict()
        for (lname, lbefore, lafter) in zipsame(['J', 'KL', 'entropy'], losses_before, losses_after):
            info[lname + "_before"] = lbefore
            info[lname + "_after"]  = lafter

        # add the baseline loss into full information
        info['B_loss'] = L
        return info


    @staticmethod
    def linesearch(f, x, fullstep, expected_improve_rate, max_backtracks=10, accept_ratio=.1):
        """
        Backtracking linesearch, where expected_improve_rate is the slope dy/dx at the initial point
        """
        fval = f(x)
        print "fval before", fval
        for (_n_backtracks, stepfrac) in enumerate(.5 ** numpy.arange(max_backtracks)):
            xnew = x + stepfrac * fullstep
            newfval = f(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepfrac
            ratio = actual_improve / expected_improve
            print "a/e/r", actual_improve, expected_improve, ratio
            if ratio > accept_ratio and actual_improve > 0:
                print "fval after", newfval
                return True, xnew
        return False, x

    @staticmethod
    def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
        """
        Conjuctate Gradient
        """
        p = b.copy()
        r = b.copy()
        x = numpy.zeros_like(b)
        rdotr = r.dot(r)

        fmtstr = "%10i %10.3g %10.3g"
        titlestr = "%10s %10s %10s"
        if verbose: print titlestr % ("iter", "residual norm", "soln norm")

        for i in xrange(cg_iters):
            if callback is not None:
                callback(x)
            if verbose: print fmtstr % (i, rdotr, numpy.linalg.norm(x))
            z = f_Ax(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break

        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i + 1, rdotr, numpy.linalg.norm(x))
        return x

    # ====================================================================== #
    # Save & Load
    # ====================================================================== #

    def save(self, history, it):
        _params = OrderedDict()
        _params = unzip(self.tparams, _params)
        _params = unzip(self.tparams_b, _params)

        print 'save the policy network >> {}'.format(self.model)
        numpy.savez('%s.current' % (self.model),
                    history=history,
                    it=it,
                    **_params)
        numpy.savez('{}.iter={}'.format(self.model, it),
                    history=history,
                    it=it,
                    **_params)

    def load(self):
        if os.path.exists(self.model):
            print 'loading from the existing model (current)'

            rmodel = numpy.load(self.model)
            history = rmodel['history']
            it = rmodel['it']

            self.params    = load_params(rmodel, self.params)
            self.params_b  = load_params(rmodel, self.params_b)
            self.tparams   = init_tparams(self.params)
            self.tparams_b = init_tparams(self.params_b)

            print 'the dataset need to go over {} lines'.format(it)
            return history, it
        else:
            return [], -1




