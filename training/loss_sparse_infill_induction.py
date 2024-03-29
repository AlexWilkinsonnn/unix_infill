import os,sys,time

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F

# sparse submanifold convnet library
sys.path.append("/unix/dune/awilkinson/infill_work")
import sparseconvnet as scn
# -------------------------------------------------------------------------
# HolePixelLoss
# This loss mimics nividia's pixelwise loss for holes (L1)
# used in the infill network
# how well does the network do in dead regions?
# -------------------------------------------------------------------------

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

class SparseInfillLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=False, ignore_index=-100 ):
        super(SparseInfillLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.reduce = False
        self.size_average = size_average
        #self.mean = torch.mean.cuda()

    def forward(self,predict,adc,input):
        """
        predict: (b,1,h,w) tensor with output from logsoftmax
        adc:  (b,h,w) tensor with true adc values
        """
        _assert_no_grad(adc)
        # print "size of predict: ",predict.size()
        # print "size of adc: ",adc.size()[0]

        # want three losses: non-dead, dead w/o charge, dead w/ charge
        L1loss=torch.nn.L1Loss(self.size_average)
        nondeadweight = 1.0
        deadnochargeweight = 500.0
        deadlowchargeweight = 100.0
        deadhighchargeweight = 100.0
        deadhighestchargeweight = 100.0

        goodch = (input != 0).float()
        predictgood = goodch * predict
        adcgood = goodch* adc
        totnondead = goodch.sum().float()
        if (totnondead == 0):
                totnondead = 1.0
        nondeadloss = (L1loss(predictgood, adcgood)*nondeadweight)/totnondead

        deadch = (input.eq(0)).float()

        deadchhighestcharge = deadch * (adc.abs() > 70).float()
        predictdeadhighestcharge = predict*deadchhighestcharge
        adcdeadhighestcharge = adc*deadchhighestcharge
        totdeadhighestcharge = deadchhighestcharge.sum().float()
        if (totdeadhighestcharge == 0):
                totdeadhighestcharge = 1.0
        deadhighestchargeloss = (L1loss(predictdeadhighestcharge,adcdeadhighestcharge)*deadhighestchargeweight)/totdeadhighestcharge

        deadchhighcharge = deadch * (adc.abs() > 40).float()*(adc.abs() < 70).float()
        predictdeadhighcharge = predict*deadchhighcharge
        adcdeadhighcharge = adc*deadchhighcharge
        totdeadhighcharge = deadchhighcharge.sum().float()
        if (totdeadhighcharge == 0):
                totdeadhighcharge = 1.0
        deadhighchargeloss = (L1loss(predictdeadhighcharge,adcdeadhighcharge)*deadhighchargeweight)/totdeadhighcharge

        deadchlowcharge = deadch * (adc.abs() > 10).float()*(adc.abs() < 40).float()

        deadchnocharge = deadch * (adc.abs() < 10).float()

        diffs = (adc.roll(1) - adc.roll(-1))
        diffs[0,0] = 0
        diffs[-1,0] = 0
        deadch_ignorewireends = deadch[deadch.bool()]
        deadch_ignorewireends_context = deadch.clone()
        deadch_ignorewireends[::6000] = 0
        deadch_ignorewireends[5999::6000] = 0
        deadch_ignorewireends_context = deadch.clone()
        deadch_ignorewireends_context[deadch.bool()] = deadch_ignorewireends
        deadchlowcharge_hidden = deadch_ignorewireends_context * (adc.abs() < 10).float() * (diffs.abs() > 20).float()
        deadchnocharge -= deadchlowcharge_hidden
        deadchlowcharge += deadchlowcharge_hidden
        
        predictdeadlowcharge = predict*deadchlowcharge
        adcdeadlowcharge = adc*deadchlowcharge
        totdeadlowcharge = deadchlowcharge.sum().float()
        if (totdeadlowcharge == 0):
                totdeadlowcharge = 1.0
        deadlowchargeloss = (L1loss(predictdeadlowcharge,adcdeadlowcharge)*deadlowchargeweight)/totdeadlowcharge

        predictdeadnocharge = predict*deadchnocharge
        adcdeadnocharge = adc*deadchnocharge
        totdeadnocharge = deadchnocharge.sum().float()
        if (totdeadnocharge == 0):
                totdeadnocharge = 1.0
        deadnochargeloss = (L1loss(predictdeadnocharge,adcdeadnocharge)*deadnochargeweight)/totdeadnocharge

        totloss = nondeadloss + deadnochargeloss + deadlowchargeloss + deadhighchargeloss+deadhighestchargeloss
        return nondeadloss, deadnochargeloss, deadlowchargeloss, deadhighchargeloss,deadhighestchargeloss, totloss
