import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox,Button
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy import optimize
import os
import json

SpineWidth = 1000

class SpatialModel():
    def __init__(self,CParams,P0Params,gamma,zetaS,FParams,C0Params,tMax,xMax,stims):

        self.a1 = CParams[0]
        self.a2 = CParams[1]

        self.b1 = P0Params[0]
        self.b2 = P0Params[1]
        self.rho = P0Params[2]

        self.gamma = gamma

        self.zeta1 = zetaS[0]
        self.zeta2 = zetaS[1]
        self.nu   = FParams[0]
        self.phi  = FParams[1]

        self.lam = C0Params[0]
        self.mu  = C0Params[1]
        self.C_s = C0Params[2]
        self.C_d = C0Params[3]


        self.dx = 0.01
        self.x = np.arange(-xMax,xMax,self.dx)


        self.dt = 0.0025
        self.tMax = tMax
        self.tvec = np.arange(0,tMax+self.dt,self.dt)

        Cd = self.initC(stims)
        Pd = 1-Cd
        self.C  = np.zeros_like(self.x) 
        PMod  = np.zeros_like(self.x) 
        
        self.P  = np.zeros_like(self.x)
        self.P0 = np.ones_like(self.x)
        
        self.S  = np.ones_like(self.x)

        for xs,c in zip(stims[0],Cd):
            self.C += (self.C_s+c)*np.exp(-SpineWidth*(self.x-xs)**2)
        for xs,c in zip(stims[0],Pd):
            PMod += (c)*np.exp(-SpineWidth*(self.x-xs)**2)
        
        self.P0 = self.rho*(self.P0-PMod)

        self.C_tot  = []
        self.P_tot  = []
        self.S_tot  = []
        self.P0_tot = []

    def CRHS(self):
    
        return self.a1*self.diff2(self.C)-self.a2*self.C

    def P0RHS(self):
        
        return self.b1*self.diff2(self.P0)-self.b2*self.C*self.P0

    def PRHS(self):
        
        return  self.b1*self.diff2(self.P) + self.b2*self.C*self.P0 - self.gamma*self.P

    def SRHS(self):
        return  self.zeta1*self.C + self.P*self.zeta2*self.F(self.S)

    def F(self,K):
        return -np.tanh(self.phi*(K-self.nu))
        
    def diff2(self,y):
        
        d2C = np.zeros_like(y)
        
        d2C[1:-1] = (np.roll(y,1)[1:-1]+np.roll(y,-1)[1:-1]-2*y[1:-1])/(self.dx*self.dx)
        
        return d2C

    def initC(self,stims,lam=1):
        x_stims = stims[0]
        t_stims = stims[1]
        C_shared     = np.ones_like(x_stims).astype(np.float64)
        n_stims = len(x_stims)
        if(n_stims>1):
            for i in range(n_stims):            
                mod = 0
                for j in range(n_stims):
                    if(i==j):
                        pass
                    else:
                        mod += (abs(x_stims[i]-x_stims[j])/(1+abs(x_stims[i]-x_stims[j])))**lam
                C_shared[i] = C_shared[i]*((mod+1)/(n_stims))
        return C_shared*self.C_d

    def Simulate(self):

        for _ in np.arange(0,self.tMax+self.dt,self.dt):
            RC   = self.CRHS()
            RP0  = self.P0RHS()
            RP   = self.PRHS()
            RS   = self.SRHS()

            self.C  = self.C  + self.dt*RC
            self.P0 = self.P0 + self.dt*RP0
            self.P  = self.P  + self.dt*RP
            self.S  = self.S  + self.dt*RS

            self.C_tot.append(self.C)
            self.P0_tot.append(self.P0)
            self.P_tot.append(self.P)
            self.S_tot.append(self.S)

        self.C_tot  = np.array(self.C_tot)
        self.P0_tot = np.array(self.P0_tot)
        self.P_tot  = np.array(self.P_tot)
        self.S_tot  = np.array(self.S_tot)

class SpatialModelAdj():

    def __init__(self,SM,stims,Input=None):

        self.SM = SM

        self.CAdj  = np.zeros_like(self.SM.x) 
        self.P0Adj = np.zeros_like(self.SM.x) 
        self.PAdj  = np.zeros_like(self.SM.x)
        self.SAdj  = np.zeros_like(self.SM.x) 

        self.C_tot  = []
        self.P_tot  = []
        self.S_tot  = []
        self.P0_tot = []

        self.Input  = Input
        self.stims  = stims


    def CAdjRHS(self):
        return (-self.SM.a1*self.SM.diff2(self.CAdj) + self.SM.a2*self.CAdj + self.SM.b2*self.P0*(self.P0Adj-self.PAdj)-self.SM.zeta1*self.SAdj)

    def P0AdjRHS(self):

        return -self.SM.b1*self.SM.diff2(self.P0Adj) + self.SM.b2*self.C*(self.P0Adj-self.PAdj)

    def PAdjRHS(self):

        return -self.SM.b1*self.SM.diff2(self.P0Adj) + self.SM.gamma*self.PAdj - self.SM.zeta2*self.SM.F(self.S)*self.SAdj

    def SAdjRHS(self):
        return -self.SM.zeta2*self.SAdj*self.P*self.dF(self.S)

    def dF(self,K):
        return -self.SM.phi*(1/np.cosh(self.SM.phi*(K-self.SM.nu)))**2

    def Force(self,ttrue):
        force = np.zeros_like(self.SAdj)
        kt = np.argmin(abs(self.SM.tvec-ttrue))
        x = np.round(self.SM.x,2)
        for s1,s2 in zip(self.stims[0],self.Input[1]):
            force[x == s1] = 2*(self.SM.S_tot[kt,x==s1]-s2[np.argmin(abs(ttrue-self.Input[0]))])
        return force

    def SimulateBack(self):

        tInputs = []
        for t in self.Input[0]: tInputs.append(len(self.SM.tvec)-np.argmin(abs(self.SM.tvec-t))-1)
        for i,t in enumerate(np.arange(0,self.SM.tMax+self.SM.dt,self.SM.dt)):
            ttrue = self.SM.tMax-t
            self.S  = self.SM.S_tot[-i]
            self.P  = self.SM.P_tot[-i]
            self.P0 = self.SM.P0_tot[-i]
            self.C  = self.SM.C_tot[-i]
            RC   = self.CAdjRHS()
            RP0  = self.P0AdjRHS()
            RP   = self.PAdjRHS()
            RS   = self.SAdjRHS()
            if(i in tInputs):
                RS+=self.Force(ttrue)
            self.CAdj  = self.CAdj  - self.SM.dt*RC
            self.P0Adj = self.P0Adj - self.SM.dt*RP0
            self.PAdj  = self.PAdj  - self.SM.dt*RP
            self.SAdj  = self.SAdj  - self.SM.dt*RS

            self.C_tot.append(self.CAdj)
            self.P0_tot.append(self.P0Adj)
            self.P_tot.append(self.PAdj)
            self.S_tot.append(self.SAdj)

        self.C_tot  = np.array(self.C_tot)
        self.P0_tot = np.array(self.P0_tot)
        self.P_tot  = np.array(self.P_tot)
        self.S_tot  = np.array(self.S_tot)

class Optimizer():

    def __init__(self,SMAdj,stims):
        self.SMAdj = SMAdj

        self.SMAdj.C_tot  = self.SMAdj.C_tot[::-1]
        self.SMAdj.P0_tot = self.SMAdj.P0_tot[::-1]
        self.SMAdj.P_tot  = self.SMAdj.P_tot[::-1]
        self.SMAdj.S_tot  = self.SMAdj.S_tot[::-1]

        self.stims = stims

    def Cost(self,Input=None):
        if(Input is None):
            return 0
        else:
            tInputs = Input[0]
            sInputs = Input[1]
            x = np.round(self.SMAdj.SM.x,3)
            tvec = self.SMAdj.SM.tvec
            cost = 0

            for stim,each_input in zip(self.stims[0],sInputs):
                for t,s in zip(tInputs,each_input):
                    try:
                        tindx = np.argmin(abs(tvec-t))
                        cost += (self.SMAdj.SM.S_tot[tindx,x==stim]-s)**2
                    except:
                        import pdb;pdb.set_trace()
            return cost/(len(self.stims[0]))

    def diff2Mat(self,y):
        
        d2C = np.zeros_like(y)
        
        d2C[:,1:-1] = (np.roll(y,1,axis=-1)[:,1:-1]+np.roll(y,-1,axis=-1)[:,1:-1]-2*y[:,1:-1])/(self.SMAdj.SM.dx**2)
        
        return d2C

    def update(self):

        dxdt = self.SMAdj.SM.dt*self.SMAdj.SM.dx
        dx   = self.SMAdj.SM.dx
        dt   = self.SMAdj.SM.dt

        C  = self.SMAdj.SM.C_tot
        P  = self.SMAdj.SM.P_tot
        P0 = self.SMAdj.SM.P0_tot
        S  = self.SMAdj.SM.S_tot

        CAdj  = self.SMAdj.C_tot
        PAdj  = self.SMAdj.P_tot
        P0Adj = self.SMAdj.P0_tot
        SAdj  = self.SMAdj.S_tot

        zeta1 = self.SMAdj.SM.zeta1
        zeta2 = self.SMAdj.SM.zeta2
        phi  = self.SMAdj.SM.phi
        nu   = self.SMAdj.SM.nu
        lam  = self.SMAdj.SM.lam

        da1 =   dxdt*np.sum(CAdj*self.diff2Mat(C)) #verified
        da2 = - dxdt*np.sum(CAdj*C) #Verified

        db1 =   dxdt*np.sum(P0Adj*self.diff2Mat(P0)+PAdj*self.diff2Mat(P)) #verified
        db2 = - dxdt*np.sum((P0Adj-PAdj)*P0*C) #verified

        dg  = - dxdt*np.sum(PAdj*P) #verified

        dzeta1 = dxdt*np.sum(C*SAdj) #verified

        dzeta2 = dxdt*np.sum(P*self.SMAdj.SM.F(S)*SAdj) #verified

        dnu  =  dxdt*zeta2*phi*np.sum((P*SAdj)/(np.cosh(phi*(S-nu))**2)) #verified

        dphi = -dxdt*zeta2*np.sum((P*SAdj*(S-nu))/(np.cosh(phi*(S-nu))**2)) #verified

        drho = dx*np.sum(P0Adj[0]*self.P0[0]) #verified

        initStim_s = np.zeros_like(C[0])
        initStim_d = np.zeros_like(C[0])

        n = len(self.stims[0])
        for i,x1 in enumerate(self.stims[0]):
            initStim_s += np.exp(-SpineWidth*(self.SMAdj.SM.x-x1)**2)
            mod  = 0
            mod2 = 0
            for j,x2 in enumerate(self.stims[0]):
                if(i==j):
                    pass
                else:
                    mod += (abs(x_stims[i]-x_stims[j])/(1+abs(x_stims[i]-x_stims[j])))**lam

            initStim_d += np.exp(-SpineWidth*(self.SMAdj.SM.x-x1)**2)*((mod+1)/(n_stims))


        dCs = dx*np.sum(CAdj[0]*initStim_s) #verified
        dCd = dx*np.sum(CAdj[0]*initStim_d) #verified
        dCl = 0

        #[0,da2,0,db2,drho,dg,dnu,dphi,dCl,0,dCs,dCd,dzeta]
        #np.array([da1,da2,db1,db2,drho,dg,dnu,dphi,dCl,0,dCs,dCd,dzeta1,dzeta2])/dt
        return np.array([da1,da2,db1,db2,drho,dg,dnu,dphi,dCl,0,dCs,dCd,dzeta1,dzeta2])/dxdt

class SingleStepper():
    def __init__(self,stims,Input,tMax,xMax,optchoice):

        self.stims = stims
        self.Input = Input
        self.tMax  = tMax
        self.xMax  = xMax

        self.optchoice = optchoice

    def SingleStep(self,params):
        cPar  = [params[0],params[1]]
        p0Par = [params[2],params[3],params[4]]
        gamma = params[5]
        fPar  = [params[6],params[7]]
        c0Par = [params[8],params[9],params[10],params[11]]
        zetaS = [params[12],params[13]]

        tMax  = 40
        xMax  = 2

        print('='*20)
        print('Params = ',params)
        fullCost = 0
        fullgrad = np.zeros_like(params)
        for s,I in zip(self.stims,self.Input):
            SM = SpatialModel(cPar,p0Par,gamma,zetaS,fPar,c0Par,self.tMax,self.xMax,s)
            SM.Simulate()

            SMA = SpatialModelAdj(SM,s,I)
            SMA.SimulateBack()

            Opt = Optimizer(SMA,s)
            fullCost += Opt.Cost(I)

            fullgrad += Opt.update()*self.optchoice
            print('~'*5)
            print('Grad   = ',Opt.update().round(5)*self.optchoice)
            print('~'*5)

        fullgrad = fullgrad/len(self.stims)


        print('Cost   = ',fullCost)
        #print('Grad   = ',grad)
        print('='*20)

        return fullCost,-np.array(fullgrad)

class VideoMaker():

    def __init__(self,SM,stimloc,RealDat=None):
        self.SM = SM
        self.stimloc = stimloc
        self.fig,ax = plt.subplot_mosaic([['A', 'B', 'B'],
                                      ['C', 'D', 'E'],
                                      ['F', 'F', 'G']],figsize=(12.,6))
        # Set the x and y data
        plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
        # Initialize the line object
        ax['A'].plot(SM.x,SM.C_tot[0])
        self.line, = ax['B'].plot(SM.x,SM.C_tot[0])
        self.line2, = ax['C'].plot(SM.x,SM.P0_tot[0])
        self.line3, = ax['D'].plot(SM.x,SM.P_tot[0])
        self.line6, = ax['E'].plot(SM.x,SM.F(SM.S_tot[0]))
        self.line4, = ax['F'].plot(SM.x,SM.S_tot[0])
        ax['F'].axvline(0,color='r',ls='--')
        self.line5, = ax['G'].plot([],[],'r-')
        if(not RealDat is None):
            ax['G'].boxplot(RealDat.T,positions=[2,10,20,30,40],showmeans=True)

        ax['A'].set_ylim([SM.C_tot.min(),SM.C_tot.max()])
        ax['B'].set_ylim([SM.C_tot.min(),SM.C_tot.max()])
        ax['C'].set_ylim([SM.P0_tot.min(),SM.P0_tot.max()])
        ax['D'].set_ylim([SM.P_tot.min(),SM.P_tot.max()])
        ax['E'].set_ylim([-2,2])
        ax['F'].set_ylim([SM.S_tot.min(),SM.S_tot.max()])
        if(not RealDat is None):
            ax['G'].set_ylim([RealDat.min()-0.1,RealDat.max()])
        else:
            ax['G'].set_ylim([1,SM.S_tot.max()+0.2])
        ax['G'].set_xlim([-2,42])

        ax['A'].set_title('Initial C')
        ax['B'].set_title('C')
        ax['C'].set_title('P_0')
        ax['D'].set_title('P')
        ax['E'].set_title('F(S)')
        ax['F'].set_title('S')
        ax['G'].set_title('S at stimulation')

    def update(self,frame):
        # Update the y data
        frameskip = int(len(self.SM.tvec)/40)
        self.line.set_ydata(self.SM.C_tot[frame*frameskip])
        self.line2.set_ydata(self.SM.P0_tot[frame*frameskip])
        self.line3.set_ydata(self.SM.P_tot[frame*frameskip])
        self.line4.set_ydata(self.SM.S_tot[frame*frameskip])
        self.line5.set_data(self.SM.tvec[:frame*frameskip],self.SM.S_tot[:frame*frameskip,self.stimloc])
        self.line6.set_ydata(self.SM.F(self.SM.S_tot[frame*frameskip]))
        return self.line, self.line2, self.line3, self.line4, self.line5, self.line6

    def Run(self):
        self.anim = FuncAnimation(self.fig, self.update, frames=40, interval=50, blit=True)
        plt.show()

def PreSortData(RealDat,Flag):
    
    Pot = []
    for d in RealDat:
        if(abs((d[3]-d[:3].mean())/d[:3].std())>1.96 and d[3]-d[:3].mean()>0):
            Pot.append(Flag)
        else:
            Pot.append(not Flag)

    return np.delete(RealDat,Pot,axis=0)

def LoadThreeSpine(Dir):
    Syn_a_arr = []
    for d in tqdm(os.listdir(Dir)):
        try:
            with open(Dir+d+'/Spine/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        except Exception as e:
            print(e)

    dists = []
    means = []
    lmeans = []
    for Syns in Syn_a_arr:
        d = []
        m = []
        lm = []
        for S in Syns:
            d.append(S["distance"])
            m.append(S["mean"])
            lm.append(S["local_bg"])
        dists.append(d)
        means.append(m)
        lmeans.append(lm)
    dists = np.array(dists)
    means = np.array(means)
    lmeans = np.array(lmeans)

    means = np.squeeze(means)
    lmeans = np.squeeze(lmeans)
    means = means-lmeans

    middle = [means[i,np.argsort(dists,axis=1)[i,1]] for i in range(7)]
    lower = [means[i,np.argsort(dists,axis=1)[i,0]] for i in range(7)]
    upper = [means[i,np.argsort(dists,axis=1)[i,2]] for i in range(7)]
    middle = np.array(middle)
    extreme = np.vstack([lower,upper])
    middleS = PreSortData(middle,False)
    middleS = (middleS.T/(middleS[:,:3].mean(axis=-1)))
    extremeS = PreSortData(extreme,False)
    extremeS = (extremeS.T/(extremeS[:,:3].mean(axis=-1)))

    return [middleS,extremeS],(dists.T-dists.min(axis=-1)).T

def LoadOneSpine(Dir):

    Syn_a_arr = []
    for d in tqdm(os.listdir(Dir)):
        print(d)
        try:
            with open(Dir+d+'/Spine/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        except Exception as e:
            print(e)
            
    dists = []
    means = []
    lmeans = []
    for Syns in Syn_a_arr:
        d = []
        m = []
        lm = []
        for S in Syns:
            d.append(S["distance"])
            m.append(S["mean"])
            lm.append(S["local_bg"])
        dists.append(d)
        means.append(m)
        lmeans.append(lm)
    dists = np.array(dists)
    means = np.array(means)
    lmeans = np.array(lmeans)
    means = np.squeeze(means)
    lmeans = np.squeeze(lmeans)
    meansS = means-lmeans

    #meansS = PreSortData(means,False)
    meansS = (meansS.T/(meansS[:,:3].mean(axis=-1)))

    return meansS

def LoadSevenSpine(Dir):
    Syn_a_arr = []
    for d in tqdm(os.listdir(Dir)):
        print(d)
        try:
            with open(Dir+d+'/Spine/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        except Exception as e:
            print(e)
            
    dists = []
    means = []
    lmeans = []
    for Syns in Syn_a_arr:
        d = []
        m = []
        lm = []
        for S in Syns:
            d.append(S["distance"])
            m.append(S["mean"])
            lm.append(S["local_bg"])
        dists.append(d)
        means.append(np.array(m)-np.array(lm))

    middle = [np.array(m)[np.argsort(d)[3]] for m,d in zip(means,dists)]
    middle = np.vstack(middle).squeeze()
    middleS = PreSortData(middle,False)
    middleS = (middleS.T/(middleS[:,:3].mean(axis=-1)))

    extreme = [np.array(m)[np.argsort(d)[[0,-1]]] for m,d in zip(means,dists)]
    extreme = np.vstack(extreme).squeeze()
    extremeS = PreSortData(extreme,False)
    extremeS = (extremeS.T/(extremeS[:,:3].mean(axis=-1)))

    return [middleS,extremeS],[np.array(d)-min(d) for d in dists]

def LoadSevenSpine2(Dir):
    Syn_a_arr = []
    for d in tqdm(os.listdir(Dir)):
        print(d)
        try:
            with open(Dir+d+'/Spine/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        except Exception as e:
            print(e)
            
    dists = []
    means = []
    lmeans = []
    for Syns in Syn_a_arr:
        d = []
        m = []
        lm = []
        for S in Syns:
            d.append(S["distance"])
            m.append(S["mean"])
            lm.append(S["local_bg"])
        dists.append(d)
        means.append(np.array(m)-np.array(lm))

    middle = [np.array(m)[np.argsort(d)[[2,3,4]]] for m,d in zip(means,dists)]
    middleS = np.vstack(middle).squeeze()
    middleS = PreSortData(middleS,False)
    middleS = (middleS.T/(middleS[:,:3].mean(axis=-1)))

    extreme = [np.array(m)[np.argsort(d)[[0,1,-2,-1]]] for m,d in zip(means,dists)]
    extremeS = np.vstack(extreme).squeeze()
    extremeS = PreSortData(extremeS,False)
    extremeS = (extremeS.T/(extremeS[:,:3].mean(axis=-1)))

    return [middleS,extremeS],[np.array(d)-min(d) for d in dists]

def Load15Spine(Dir):
    Syn_a_arr = []
    for d in tqdm(os.listdir(Dir)):
        print(d)
        try:
            with open(Dir+d+'/Spine/Synapse_l.json', 'r') as fp: Syn_a_arr.append(json.load(fp))
        except Exception as e:
            print(e)
            
    dists = []
    means = []
    lmeans = []
    for Syns in Syn_a_arr:
        d = []
        m = []
        lm = []
        for S in Syns:
            d.append(S["distance"])
            m.append(S["mean"])
            lm.append(S["local_bg"])
        dists.append(d)
        means.append(np.array(m)-np.array(lm))

    middle = [np.array(m)[np.argsort(d)[7]] for m,d in zip(means,dists)]
    middle = np.vstack(middle).squeeze()
    middleS = PreSortData(middle,False)
    middleS = (middleS.T/(middleS[:,:3].mean(axis=-1)))

    sub_middle = [np.array(m)[np.argsort(d)[[2,3]]] for m,d in zip(means,dists)]
    sub_middle = np.vstack(sub_middle).squeeze()
    sub_middleS = PreSortData(sub_middle,False)
    sub_middleS = (sub_middleS.T/(sub_middleS[:,:3].mean(axis=-1)))

    extreme = [np.array(m)[np.argsort(d)[[0,-1]]] for m,d in zip(means,dists)]
    extreme = np.vstack(extreme).squeeze()
    extremeS = PreSortData(extreme,False)
    extremeS = (extremeS.T/(extremeS[:,:3].mean(axis=-1)))

    return [middleS,sub_middleS,extremeS],[np.array(d)-min(d) for d in dists]

def Opt(params,optchoice):


    #DataDir = './3Spine/'
    #DataDir2 = './1SpineCam/'
    #DataDir7 = './7Spine/'
    #DataDir15 = './15Spine/'
    #m,d = LoadThreeSpine(DataDir)
    #m2  = LoadOneSpine(DataDir2)
    #m7,d7 = LoadSevenSpine2(DataDir7)
    m15,d15 = Load15Spine(DataDir15)
    #stims_one   = np.array([[0],[0]])
    #stims_three = np.array([[-0.19,0,0.19],[0,0,0]])
    #stims_seven   = np.array([[-0.63,-0.42,-0.21,0,0.21,0.42,0.63],[0,0,0,0,0,0,0]])
    #stims_seven   = np.array([[-1.89,-1.26,-0.63,0,0.63,1.26,1.89],[0,0,0,0,0,0,0]])
    stims_fifteen = np.array([[-1.68,-1.44,-1.2,-0.96,-0.72,-0.48,-0.24,0,0.24,0.48,0.72,0.96,1.2,1.44,1.68],
                            [0]*15])

    #Test

    #params = params+np.random.rand(14)

    a1 = params[0]
    a2 = params[1]

    b1 = params[2]
    b2 =  params[3]
    rho = params[4]
    gamma = params[5]

    nu = params[6]
    phi = params[7]

    lam = 1#params[8]
    mu  = 1#params[9] 
    Cs  = params[10]
    Cd  = params[11]

    zeta1 = params[12]
    zeta2 = params[13]

    tMax  = 40
    xMax  = 2

    times = [2,10,20,30,40]

    SM1 = SpatialModel([a1,a2],[b1,b2,rho],gamma,[zeta1,zeta2],[nu,phi],[lam,mu,Cs,Cd],tMax,xMax,stims_one)
    SM1.Simulate()
    xvec = SM1.x.round(2)
    stimlocs_one = np.argwhere(xvec == 0)[0][0]
    stimlocs_three = [np.argwhere(xvec == s)[0,0] for s in stims_three[0]]
    stimlocs_seven = [np.argwhere(xvec == s)[0,0] for s in stims_seven[0]]


    SM2 = SpatialModel([a1,a2],[b1,b2,rho],gamma,[zeta1,zeta2],[nu,phi],[lam,mu,Cs,Cd],tMax,xMax,stims_three)
    SM2.Simulate()

    SM3 = SpatialModel([a1,a2],[b1,b2,rho],gamma,[zeta1,zeta2],[nu,phi],[lam,mu,Cs,Cd],tMax,xMax,stims_seven)
    SM3.Simulate()
    VM = VideoMaker(SM3,stimlocs_one)
    VM.Run()

    SM4 = SpatialModel([a1,a2],[b1,b2,rho],gamma,[zeta1,zeta2],[nu,phi],[lam,mu,Cs,Cd],tMax,xMax,stims_fifteen)
    SM4.Simulate()


    try:
        fig,ax = plt.subplots(1,4,figsize=(36,9))
        ax[0].boxplot(m2[3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[0].plot(SM1.tvec,SM1.S_tot[:,stimlocs_one])
        ax[0].axvline(2,ls = '--',alpha=0.25)
        ax[0].axvline(10,ls = '--',alpha=0.25)
        ax[0].axvline(20,ls = '--',alpha=0.25)
        ax[0].axvline(30,ls = '--',alpha=0.25)
        ax[0].axvline(40,ls = '--',alpha=0.25)
        ax[0].set_ylim([0,m[0][3:].max()])

        ax[1].boxplot(m[1][3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[1].plot(SM2.tvec,SM2.S_tot[:,stimlocs_one])
        ax[1].axvline(2,ls = '--',alpha=0.25)
        ax[1].axvline(10,ls = '--',alpha=0.25)
        ax[1].axvline(20,ls = '--',alpha=0.25)
        ax[1].axvline(30,ls = '--',alpha=0.25)
        ax[1].axvline(40,ls = '--',alpha=0.25)
        ax[1].set_ylim([0,m[0][3:].max()])

        ax[2].boxplot(m7[1][3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[2].plot(SM3.tvec,SM3.S_tot[:,stimlocs_one])
        ax[2].axvline(2,ls = '--',alpha=0.25)
        ax[2].axvline(10,ls = '--',alpha=0.25)
        ax[2].axvline(20,ls = '--',alpha=0.25)
        ax[2].axvline(30,ls = '--',alpha=0.25)
        ax[2].axvline(40,ls = '--',alpha=0.25)
        ax[2].set_ylim([0,m[0][3:].max()])

        ax[3].boxplot(m15[-1][3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[3].plot(SM3.tvec,SM4.S_tot[:,stimlocs_one])
        ax[3].axvline(2,ls = '--',alpha=0.25)
        ax[3].axvline(10,ls = '--',alpha=0.25)
        ax[3].axvline(20,ls = '--',alpha=0.25)
        ax[3].axvline(30,ls = '--',alpha=0.25)
        ax[3].axvline(40,ls = '--',alpha=0.25)
        ax[3].set_ylim([0,m[0][3:].max()])
        plt.show()
    except:
        import pdb;pdb.set_trace()

    times = [2,10,20,30,40]


    #Inputs_one   = [times,[m2.mean(axis=-1)[3:]]]
    #Inputs_three = [times,[m[1].mean(axis=-1)[3:],m[0].mean(axis=-1)[3:],m[1].mean(axis=-1)[3:]]]
    #Inputs_seven = [times,[m7[1].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:]]]
    Inputs_seven = [times,[m7[1].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[0].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:],m7[1].mean(axis=-1)[3:]]]
    
    #SS = SingleStepper([stims_one],[Inputs_one],tMax,xMax,optchoice) #one
    #SS = SingleStepper([stims_three],[Inputs_three],tMax,xMax,optchoice) #three
    SS = SingleStepper([stims_seven],[Inputs_seven],tMax,xMax,optchoice) #seven
    
    boundvec =  np.array([[0,np.inf]]*14)
    boundvec[0] = [0.0000001,0.01]
    boundvec[2] = [0.0000001,0.01]
    #boundvec[4] = [0,1.41171983e-01]
    #boundvec[-4] = [0,9.44142944e-01]
    #boundvec[-3] = [0,1.21359463e+00]
    #boundvec[6] = [1,np.inf]
    res = optimize.minimize(SS.SingleStep, np.array([a1,a2,b1,b2,rho,gamma,nu,phi,lam,mu,Cs,Cd,zeta1,zeta2]),bounds=boundvec, options={'ftol':1e-5,'gtol': 1e-5,'disp':True,'maxls':20},jac=True,method='L-BFGS-B',tol=1e-5)

    a1,a2,b1,b2,rho,gamma,nu,phi,lam,mu,Cs,Cd,zeta1,zeta2 = res.x

    print('Params = ',res.x.round(3))

    cPar  = [a1,a2]
    p0Par = [b1,b2,rho]
    gamma = gamma
    fPar  = [nu,phi]
    c0Par = [lam,mu,Cs,Cd]
    zetaS  = [zeta1,zeta2]
    
    SM1 = SpatialModel(cPar,p0Par,gamma,zetaS,fPar,c0Par,tMax,xMax,stims_one)
    SM1.Simulate()
    VM = VideoMaker(SM1,stimlocs_one)
    VM.Run()

    SM2 = SpatialModel(cPar,p0Par,gamma,zetaS,fPar,c0Par,tMax,xMax,stims_three)
    SM2.Simulate()
    VM = VideoMaker(SM2,stimlocs_one)
    VM.Run()
    
    SM3 = SpatialModel([a1,a2],[b1,b2,rho],gamma,[zeta1,zeta2],[nu,phi],[lam,mu,Cs,Cd],tMax,xMax,stims_seven)
    SM3.Simulate()
    VM = VideoMaker(SM3,stimlocs_one)
    VM.Run()

    try:
        fig,ax = plt.subplots(1,3,figsize=(27,9))
        ax[0].boxplot(m2[3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[0].plot(SM1.tvec,SM1.S_tot[:,stimlocs_one])
        ax[0].axvline(2,ls = '--',alpha=0.25)
        ax[0].axvline(10,ls = '--',alpha=0.25)
        ax[0].axvline(20,ls = '--',alpha=0.25)
        ax[0].axvline(30,ls = '--',alpha=0.25)
        ax[0].axvline(40,ls = '--',alpha=0.25)
        ax[0].set_ylim([0,m[0][3:].max()])

        ax[1].boxplot(m[0][3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[1].plot(SM2.tvec,SM2.S_tot[:,stimlocs_one])
        ax[1].axvline(2,ls = '--',alpha=0.25)
        ax[1].axvline(10,ls = '--',alpha=0.25)
        ax[1].axvline(20,ls = '--',alpha=0.25)
        ax[1].axvline(30,ls = '--',alpha=0.25)
        ax[1].axvline(40,ls = '--',alpha=0.25)
        ax[1].set_ylim([0,m[0][3:].max()])

        ax[2].boxplot(m7[0][3:].T,positions=[2,10,20,30,40],showmeans=True)
        ax[2].plot(SM3.tvec,SM3.S_tot[:,stimlocs_one])
        ax[2].axvline(2,ls = '--',alpha=0.25)
        ax[2].axvline(10,ls = '--',alpha=0.25)
        ax[2].axvline(20,ls = '--',alpha=0.25)
        ax[2].axvline(30,ls = '--',alpha=0.25)
        ax[2].axvline(40,ls = '--',alpha=0.25)
        ax[2].set_ylim([0,m[0][3:].max()])
        plt.show()
    except:
        import pdb;pdb.set_trace()

    plt.boxplot(m[1][3:].T,positions=[2,10,20,30,40])
    plt.plot(SM2.tvec,SM3.S_tot[:,stimlocs_three[1]])
    plt.axvline(2,ls = '--',alpha=0.25)
    plt.axvline(10,ls = '--',alpha=0.25)
    plt.axvline(20,ls = '--',alpha=0.25)
    plt.axvline(30,ls = '--',alpha=0.25)
    plt.axvline(40,ls = '--',alpha=0.25)
    plt.show()

    plt.plot(SM2.tvec,SM3.S_tot[:,stimlocs_three[0]],'b')
    plt.plot(SM2.tvec,SM3.S_tot[:,stimlocs_three[1]],'r')
    plt.show()

if __name__ == '__main__':


    #params = [1.74092432e-03 ,1.05321253e+00, 4.38247049e-03, 1.20882739e+00,
 #1.24707992e-01 ,2.77111665e-02 ,1.27804292e+00 ,2.01189067e+00,
# 9.81004876e-01 ,1.00000000e+00 ,9.44079964e-01, 1.21356463e+00,
# 3.01822565e+00 ,7.94478455e+00] # 3.2 works with rho and with C


    params = np.load('./ModelFit/ControlFit3.npy')

    optchoice = np.array([0,0,0,0,1,0,0,0,0,0,0,1,0,0])

    Opt(params,optchoice)

    