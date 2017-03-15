"""

The log-bilinear language model from (Mnih and Teh, ICML 2012)

References:
 A fast and simple algorithm for training neural probabilistic language models. 
 Andriy Mnih and Yee Whye Teh.
 International Conference on Machine Learning 2012 (ICML 2012) 

Usage: lbl.py [--verbose] [--word_dim WORD_DIM] [--context_sz CONTEXT_SZ] 
              [--learn_rate LEARN_RATE] [--rate_update RATE_UPDATE] 
              [--epochs EPOCHS] [--batch_size BATCH_SZ] [--seed SEED]  
              [--patience PATIENCE] [--patience_incr PATIENCE_INCR] 
              [--improvement_thrs IMPR_THRS] [--validation_freq VALID_FREQ] 
              [--model MODEL_FILE] 
              <train_data> <dev_data> [<test_data>]

Arguments:
 train_data       training data of tokenized text, one sentence per line.
 dev_data         development data of tokenized text, one sentence per line.
 test_data        test data of tokenized text, one sentence per line.

Options:
    -v, --verbose                                Print debug information
    -k WORD_DIM, --word_dim=WORD_DIM             dimension of word embeddings [default: 100]
    -n CONTEXT_SZ, --context_sz=CONTEXT_SZ       size of n-gram context window [default: 2]
    -l LEARN_RATE, --learn_rate=LEARN_RATE       initial learning rate parameter [default: 1]
    -u RATE_UPDATE, --rate_update=RATE_UPDATE    learning rate update: 'simple', 'adaptive' [default: simple]
    -e EPOCHS, --epochs=EPOCHS                   number of training epochs [default: 10]
    -b BATCH_SIZE, --batch_size=BATCH_SIZE       size of mini-batch for training [default: 100]
    -s SEED, --seed=SEED                         seed for random generator.
    -p PATIENCE, --patience PATIENCE             min number of examples to look before stopping, default is no early stopping
    -i PATIENCE_INCR, --patience_incr=PATIENCE   wait for this much longer when a new best result is found [default: 2]
    -t IMPR_THRS, --improvement_thrs=IMPR_THRS   a relative improvemnt of this is considered significant [default: 0.995]
    -f VALID_FREQ, --validation_freq=VALID_FREQ  number of examples after which check score on dev set [default: 1000]
    -m MODEL_FILE, --model=MODEL_FILE            file to save the model to. Default none.
"""

from dataset import Dictionary, load_corpus

import cPickle
import numpy as np
import theano
import theano.tensor as T
import math
from future_builtins import zip
import time
import logging

logger = logging.getLogger(__name__)

theano.config.blas.ldflags = "-L/export/apps/lib64/atlas/ " + theano.config.blas.ldflags
theano.config.exception_verbosity = "high"

class LogBilinearLanguageModel(object):
    """
    Log-bilinear language model class
    """

    def __init__(self, context, V, K, num_sub_tags, feature_matrix_values, context_sz, rng):
        """
        Initialize the parameters of the language model
        """
        # training contexts
        self.context = context
       
        # initialize context word embedding matrix R of shape (V, K)
        # TODO: parameterize initialization
        R_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)
        R_values[:,0:2] = np.zeros((V,2))
        self.R = theano.shared(value=R_values, name='R', borrow=True)
        # initialize target word embedding matrix Q of shape (V, K)
        Q_values = np.asarray(rng.uniform(-0.01, 0.01, size=(V, K)), 
                              dtype=theano.config.floatX)

        Q_values[:,0:2] = np.zeros((V,2))
        self.Q = theano.shared(value=Q_values, name='Q', borrow=True)
        # initialize weight tensor C of shape (context_sz, K, K)
        C_values = np.asarray(rng.normal(0, math.sqrt(0.1), 
                                         size=(context_sz, K, K)), 
                              dtype=theano.config.floatX)
        self.C = theano.shared(value=C_values, name='C', borrow=True)

        # initialize tag matrix
        Tag_values = np.asarray(rng.normal(-0.01,0.01,size=(num_sub_tags,K)),
                                dtype=theano.config.floatX)
        self.Tag = theano.shared(value=Tag_values,name='Tag',borrow=True)

        # initialize bias vector 
        b_values = np.asarray(rng.normal(0, math.sqrt(0.1), size=(V,)), 
                              dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        # context word representations
        self.r_w = self.R[context]
        # predicted word representation for target word
        self.q_hat = T.tensordot(self.C, self.r_w, axes=[[0,1], [1,2]])
        # similarity score between predicted word and all target words
      
        self.s = T.transpose(T.dot(self.Q, self.q_hat) + T.reshape(self.b, (V,1)))
        # softmax activation function
        self.p_w_given_h = T.nnet.softmax(self.s)

        
        self.feature_matrix = theano.shared(value=feature_matrix_values,name="feature_matrix",borrow=True)
    

        # activation function for tags

        # feature_matrix : Tag Size x Sub Tag Size
        # Tag : Sub Tag Size x K
        # Q.T : K x V
        # s_tag = Tag Size x V 
        self.s_tag = T.dot(T.dot(self.feature_matrix,self.Tag),T.transpose(self.Q))
        #self.s_tag = T.dot((T.dot(self.feature_matrix,self.Tag)),T.transpose(self.Q))
        # softmax activation function tag given word distribution
        self.p_t_given_w = T.nnet.softmax(self.s_tag)


        # parameters of the model
        self.params = [self.R, self.Q, self.C, self.b, self.Tag]

    def negative_log_likelihood(self, y, t):
        # take the logarithm with base 2
        
        #theano.printing.Print(self.r_w)
        
        return -T.mean(T.log2(self.p_w_given_h)[T.arange(y.shape[0]), y]) - T.mean(T.log2(self.p_t_given_w)[t,y])


def make_instances(corpus, vocab, vocab_tags, context_sz, start_symb='<s>', end_symb='</s>'):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y, data_tags = data_xy
       
        shared_x = theano.shared(np.asarray(data_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=borrow)
        shared_tags = theano.shared(np.asarray(data_tags, dtype=np.int32), borrow=borrow)
        return shared_x, shared_y, shared_tags
    data = []
    labels = []
    tag_labels = []

    for sentence in corpus:
        sentence_word = map(lambda x: x[0],sentence)
        sentence_tag = map(lambda x: x[1],sentence)

        sentence = [start_symb] * context_sz + sentence_word + [end_symb] * context_sz
        sentence = vocab.doc_words_to_ids(sentence, update_dict=False)

        sentence_tag = [start_symb] * context_sz + sentence_tag + [end_symb] * context_sz
        sentence_tag = vocab_tags.doc_words_to_ids(sentence_tag,update_dict=False)
      
        for instance in zip(*(sentence[i:] for i in xrange(context_sz+1))):
            data.append(instance[:-1])
            labels.append(instance[-1])

        tag_labels.extend(sentence_tag[context_sz:])
       

        for word_i,word in enumerate(sentence):
            if word_i in [len(sentence)-1,len(sentence)-2,len(sentence)-3,len(sentence)-4]:
                continue
                
            prefixes = [0] * 5
            for prefix_i,prefix in enumerate(vocab.word_to_prefixes[vocab.lookup_word(word)]):
                if prefix == "":
                    continue

                prefix_id = vocab.prefix_to_id[prefix]
                prefixes[prefix_i-1] = (len(vocab.vocab) + 1) + prefix_id

            suffixes = [0] * 5
            for suffix_i,suffix in enumerate(vocab.word_to_suffixes[vocab.lookup_word(word)]):
                if suffix == "":
                    continue
                    
                suffix_id = vocab.suffix_to_id[suffix]
                suffixes[suffix_i] = vocab.num_prefixes + (len(vocab.vocab) + 1) +  suffix_id

    train_set_x, train_set_y, train_set_tags = shared_dataset([data, labels,tag_labels])
    return train_set_x, train_set_y, train_set_tags
        
def make_tag_features(tag_set,tag_to_sub_ids):
    def shared_dataset(data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=np.int32), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=np.int32), borrow=borrow)
        return shared_x, shared_y
    data = []
    labels = []        
    for sentence in corpus:
        sentence_word = map(lambda x: x[0],sentence)
        sentence_tag = map(lambda x: x[1],sentence)
        # add 'start of sentence' and 'end of sentence' context
        sentence = [start_symb] * context_sz + sentence_word + [end_symb] * context_sz
        sentence = vocab.doc_words_to_ids(sentence, update_dict=False)
        
        for instance in zip(*(sentence[i:] for i in xrange(context_sz+1))):
            data.append(instance[:-1])
            labels.append(instance[-1])
        
    train_set_x, train_set_y = shared_dataset([data, labels])
    return train_set_x, train_set_y


def train_lbl(train_data, dev_data, test_data=[], 
              K=20, context_sz=2, learning_rate=1.0, 
              rate_update='simple', epochs=10, 
              batch_size=1, rng=None, patience=None, 
              patience_incr=2, improvement_thrs=0.995, 
              validation_freq=1000):

    """ Train log-bilinear model """
    # create vocabulary from train data, plus <s>, </s>
    
    logger.info("Creating vocabulary dictionary...")
    vocab = Dictionary.from_corpus(train_data, unk='<unk>')
    logger.info("Creating tag dictionary...")
    vocab_tags = Dictionary.from_corpus_tags(train_data, unk='<unk>')
    vocab.add_word('<s>')
    vocab.add_word('</s>')
    V = vocab.size()

    vocab_tags.add_word('<s>')
    vocab_tags.add_word('</s>')
    V_tag = vocab_tags.size()
    #print train_data
    
    # initialize random generator if not provided
    rng = np.random.RandomState() if not rng else rng
    
    logger.info("Making instances...")
    # generate (context, target) pairs of word ids
    train_set_x, train_set_y, train_set_tags = make_instances(train_data, vocab, vocab_tags, context_sz)
    dev_set_x, dev_set_y, dev_set_tags  = make_instances(dev_data, vocab, vocab_tags, context_sz)
    test_set_x, test_set_y, test_set_tags  = make_instances(test_data, vocab, vocab_tags, context_sz)
    
    # make feature_matrix 
    # very sparse matrix...better way to do it?
    feature_matrix = np.zeros((vocab_tags.size(),vocab_tags.num_sub_tags))
    feature_matrix[(0,0)] = 1 # unk encoding
    
    for tag,tag_id in vocab_tags:
        if tag == "<s>":
            feature_matrix[(tag_id,1)] = 1
        elif tag == "</s>":
            feature_matrix[(tag_id,2)] = 1
        else:
            for sub_tag in vocab_tags.map_tag_to_sub_tags[tag]:
                val = vocab_tags.map_sub_to_ids[sub_tag]
                feature_matrix[(tag_id,val)] = 1
             
    feature_matrix[1,:] = np.zeros((vocab_tags.num_sub_tags))
    # number of minibatches for training
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_dev_batches = dev_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    # build the model
    logger.info("Build the model ...")
    index = T.lscalar()
    
    x = T.imatrix('x')
    y = T.ivector('y')
    t = T.ivector('t') # the tag vector
    
    # create log-bilinear model
    lbl = LogBilinearLanguageModel(x, V, K, vocab_tags.num_sub_tags, feature_matrix, context_sz, rng)
 

    # cost function is negative log likelihood of the training data
    cost = lbl.negative_log_likelihood(y,t)
  
    # compute the gradient
    gparams = []
    for param in lbl.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # specify how to update the parameter of the model
    updates = []
    for param_i,(param, gparam) in enumerate(zip(lbl.params, gparams)):
        updates.append((param, param-learning_rate*gparam))
                        
    # function that computes log-probability of the dev set
    logprob_dev = theano.function(inputs=[index], outputs=cost,
                                  givens={x: dev_set_x[index*batch_size:
                                                           (index+1)*batch_size],
                                          y: dev_set_y[index*batch_size:
                                                           (index+1)*batch_size],
                                          t: dev_set_tags[index*batch_size:(index+1)*batch_size]
                                          })


    # function that computes log-probability of the test set
    logprob_test = theano.function(inputs=[index], outputs=cost,
                                   givens={x: test_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                           y: test_set_y[index*batch_size:
                                                             (index+1)*batch_size],
                                           t: test_set_tags[index*batch_size:(index+1)*batch_size]
                                       })
    
    # function that returns the cost and updates the parameter 
    train_model = theano.function(inputs=[index], outputs=cost,
                                  updates=updates,
                                  givens={x: train_set_x[index*batch_size:
                                                             (index+1)*batch_size],
                                          y: train_set_y[index*batch_size:
                                                             (index+1)*batch_size],
                                          t: train_set_tags[index*batch_size:(index+1)*batch_size]
                                          })


    # perplexity functions
    def compute_dev_logp():
        return np.mean([logprob_dev(i) for i in xrange(n_dev_batches)])

    def compute_test_logp():
        return np.mean([logprob_test(i) for i in xrange(n_test_batches)])

    def ppl(neg_logp):
        return np.power(2.0, neg_logp)
    
    # train model
    logger.info("training model...")
    best_params = None
    last_epoch_dev_ppl = np.inf
    best_dev_ppl = np.inf
    test_ppl = np.inf
    test_core = 0
    start_time = time.clock()
    done_looping = False

    for epoch in xrange(epochs):
        if done_looping:
            break
        logger.info('epoch %i' % epoch) 
        for minibatch_index in xrange(n_train_batches):
            itr = epoch * n_train_batches + minibatch_index
            train_logp = train_model(minibatch_index)
            logger.info('epoch %i, minibatch %i/%i, train minibatch log prob %.4f ppl %.4f' % 
                         (epoch, minibatch_index+1, n_train_batches, 
                          train_logp, ppl(train_logp)))
            if (itr+1) % validation_freq == 0:
                # compute perplexity on dev set, lower is better
                dev_logp = compute_dev_logp()
                dev_ppl = ppl(dev_logp)
                logger.debug('epoch %i, minibatch %i/%i, dev log prob %.4f ppl %.4f' % 
                             (epoch, minibatch_index+1, n_train_batches, 
                              dev_logp, ppl(dev_logp)))
                # if we got the lowest perplexity until now
                if dev_ppl < best_dev_ppl:
                    # improve patience if loss improvement is good enough
                    if patience and dev_ppl < best_dev_ppl * improvement_thrs:
                        patience = max(patience, itr * patience_incr)
                    best_dev_ppl = dev_ppl
                    test_logp = compute_test_logp()
                    test_ppl = ppl(test_logp)
                    logger.debug('epoch %i, minibatch %i/%i, test log prob %.4f ppl %.4f' % 
                                 (epoch, minibatch_index+1, n_train_batches, 
                                  test_logp, ppl(test_logp)))
            # stop learning if no improvement was seen for a long time
            if patience and patience <= itr:
                done_looping = True
                break
        # adapt learning rate
        if rate_update == 'simple':
            # set learning rate to 1 / (epoch+1)
            learning_rate = 1.0 / (epoch+1)
        elif rate_update == 'adaptive':
            # half learning rate if perplexity increased at end of epoch (Mnih and Teh 2012)
            this_epoch_dev_ppl = ppl(compute_dev_logp())
            if this_epoch_dev_ppl > last_epoch_dev_ppl:
                learning_rate /= 2.0
            last_epoch_dev_ppl = this_epoch_dev_ppl
        elif rate_update == 'constant':
            # keep learning rate constant
            pass
        else:
            raise ValueError("Unknown learning rate update strategy: %s" %rate_update)
        
    end_time = time.clock()
    total_time = end_time - start_time
    logger.info('Optimization complete with best dev ppl of %.4f and test ppl %.4f' % 
                (best_dev_ppl, test_ppl))
    logger.info('Training took %d epochs, with %.1f epochs/sec' % (epoch+1, 
                float(epoch+1) / total_time))
    logger.info("Total training time %d days %d hours %d min %d sec." % 
                (total_time/60/60/24, total_time/60/60%24, total_time/60%60, total_time%60))
    # return model
    return lbl

    
if __name__ == '__main__':
    import sys
    from docopt import docopt

    # parse command line arguments
    arguments = docopt(__doc__)
    log_level= logging.DEBUG if arguments['--verbose'] else logging.INFO
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)
    logger.setLevel(log_level)
    word_dim = int(arguments['--word_dim'])
    context_sz = int(arguments['--context_sz'])
    learn_rate = float(arguments['--learn_rate'])
    rate_update = arguments['--rate_update']
    epochs = int(arguments['--epochs'])
    batch_sz = int(arguments['--batch_size'])
    seed = int(arguments['--seed']) if arguments['--seed'] else None
    patience = int(arguments['--patience']) if arguments['--patience'] else None
    patience_incr = int(arguments['--patience_incr'])
    improvement_thrs = float(arguments['--improvement_thrs'])
    validation_freq = int(arguments['--validation_freq'])
    outfile = arguments['--model']

    # load data
    logger.info("Load data ...")
    
    def read_in_data(file_in):
        # ryan's reading in of the morphologically rich data
        
        counter = 0
        data = []
        with open(file_in, 'rb') as fin:
            cur_sentence = []
            for line in fin.readlines():
                line = line.rstrip("\n")
                
                if line == "":
                    # line break
                    data.append(cur_sentence)
                    cur_sentence = []
                else:
                    word,tag = line.split("\t")
                    cur_sentence.append((word,tag))
            
            
            data.append(cur_sentence)
            
        return data
    logger.info("Reading in training data...")
    train_data = read_in_data(arguments['<train_data>'])
    logger.info("Reading in dev data...")
    dev_data = read_in_data(arguments['<dev_data>'])
    logger.info("Reading in test data...")
    test_data = read_in_data(arguments['<test_data>'])

    # create random number generator
    rng_state = np.random.RandomState(seed)
    
    logger.info("Creating model...")
    # train lm model
    lbl = train_lbl(train_data, dev_data, test_data=test_data, 
              K=word_dim, context_sz=context_sz, learning_rate=learn_rate, 
              rate_update=rate_update, epochs=epochs, batch_size = batch_sz, 
              rng=rng_state, patience=patience, patience_incr=patience_incr, 
              improvement_thrs=improvement_thrs, validation_freq=validation_freq)
    
    # save the model
    if outfile:
        logger.info("Saving model ...")
        
        for param in lbl.params:
            fout = open(outfile + str(param), 'wb')
            cPickle.dump(param.get_value(borrow=True), fout, -1)
            fout.close()
        
        
