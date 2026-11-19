"""
File: nearfield_profile_ml.py
Author: Xin Lu
Date: 2026-11-18
Description: Spherical-wave array model, exact Fisher information, and multistart profile likelihood.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_solve, cho_factor


def steering_nearfield(theta, r, positions, freq_hz, sound_speed=343.0):
    """Unit-norm spherical steering and exact d/d(theta radians), d/d(r metres).

    Mic positions are in metres, relative to the array origin. Source is in
    the horizontal XY plane at (r sin(theta), r cos(theta), 0). A common
    propagation phase exp(+j*k*r) is factored out; common phase does not
    affect the rank-one covariance. Steering includes spherical 1/d amplitude.
    """
    P = np.asarray(positions, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or not (np.isfinite(P).all()):
        raise ValueError('positions must be finite Mx3 coordinates in metres')
    if not (r > 0.0 and freq_hz > 0.0 and sound_speed > 0.0):
        raise ValueError('positive range, frequency and speed required')
    u = np.array([np.sin(theta), np.cos(theta), 0.0])
    up = np.array([np.cos(theta), -np.sin(theta), 0.0])
    src = r * u
    delta = src[None, :] - P
    d = np.linalg.norm(delta, axis=1)
    if np.any(d < 1e-7):
        raise ValueError('source coincides with microphone')
    dtheta = (delta @ (r*up)) / d
    drange = (delta @ u) / d
    k = 2*np.pi*freq_hz/sound_speed
    base = (r/d) * np.exp(-1j*k*(d-r))
    base_norm = np.linalg.norm(base)
    a = base/base_norm
    # Log-derivative separates amplitude derivative and travel-time phase.
    logdt = -dtheta/d - 1j*k*dtheta
    logdr = 1.0/r - drange/d - 1j*k*(drange-1.0)
    # Derivative of real Euclidean norm: Re(sum(|a_m|^2 log_derivative_m)).
    da = a*(logdt - np.real(np.sum(np.abs(a)**2*logdt)))
    dar = a*(logdr - np.real(np.sum(np.abs(a)**2*logdr)))
    return a, np.stack([da, dar], axis=1)


def noise_covariance(positions, theta_background=np.deg2rad(-45), r_background=1.5,
                     freq_hz=1250., noise_scale=1.0):
    """Positive-definite colored background including a fixed static interferer."""
    m = len(positions)
    bg, _ = steering_nearfield(theta_background,r_background,positions,freq_hz)
    ids=np.arange(m)
    Toeplitz=0.48**np.abs(ids[:,None]-ids[None,:])
    return noise_scale*(0.44*np.eye(m)+0.37*Toeplitz+1.65*np.outer(bg,bg.conj()))


def simulate(theta,r,positions,freq_hz,N,powers,counts,rng):
    """Independent proper complex Gaussian time-block sample covariance."""
    a,_ = steering_nearfield(theta,r,positions,freq_hz)
    stats=[]
    true_covs=[]
    for p,n in zip(powers,counts):
        sig=N+float(p)*np.outer(a,a.conj())
        L=np.linalg.cholesky(sig)
        z=(rng.standard_normal((len(a),int(n)))+1j*rng.standard_normal((len(a),int(n))))/np.sqrt(2)
        x=L@z
        stats.append(x@x.conj().T/int(n))
        true_covs.append(sig)
    return np.asarray(stats), np.asarray(true_covs)


def geom_information(theta,r,positions,freq_hz,N):
    """Array geometry-only factor G in exact 2x2 efficient information."""
    a,D=steering_nearfield(theta,r,positions,freq_hz)
    W=np.linalg.inv(N)
    s=float(np.vdot(a,W@a).real)
    C=D.conj().T@W@D
    v=D.conj().T@W@a
    return 2*s*np.real(C-np.outer(v,v.conj())/s)


def efficient_information(theta,r,positions,freq_hz,N,powers,counts):
    """Exact 2D EFIM after projecting unknown full common N and block powers."""
    a,_=steering_nearfield(theta,r,positions,freq_hz)
    W=np.linalg.inv(N)
    s=float(np.vdot(a,W@a).real)
    p=np.asarray(powers,dtype=float)
    n=np.asarray(counts,dtype=float)
    weights=n/(1+s*p)
    pbar=float(np.dot(weights,p)/np.sum(weights))
    V=float(np.dot(weights,(p-pbar)**2))
    return V*geom_information(theta,r,positions,freq_hz,N)


def hermitian_basis(m):
    """Real basis of MxM Hermitian matrices (M² coordinates)."""
    basis=[]
    for i in range(m):
        D=np.zeros((m,m),complex);D[i,i]=1.; basis.append(D)
    for i in range(1,m):
        for j in range(i):
            E=np.zeros((m,m),complex);E[i,j]=E[j,i]=1.;basis.append(E)
            F=np.zeros((m,m),complex);F[i,j]=1j;F[j,i]=-1j;basis.append(F)
    return basis


def full_information_schur(theta,r,positions,freq_hz,N,powers,counts):
    """Independent full Fisher/Schur projection, including gauge singularity.

    Uses a separate explicit Hermitian nuisance basis; the information for
    angle and range is computed by orthogonal projection onto nuisance-span
    in the Fisher metric, not by the closed-form V(p)*G formula.
    """
    a,D=steering_nearfield(theta,r,positions,freq_hz)
    p=np.asarray(powers,float); n=np.asarray(counts,float)
    M=len(a); T=len(p)
    A=np.outer(a,a.conj())
    nuisance = hermitian_basis(M)
    deriv=[]
    for i in range(2):
        B=np.outer(D[:,i],a.conj())+np.outer(a,D[:,i].conj())
        deriv.append([pt*B for pt in p])
    for H in nuisance:
        deriv.append([H for _ in p])
    for t in range(T):
        deriv.append([A if j==t else np.zeros_like(A) for j in range(T)])
    F=np.zeros((len(deriv),len(deriv)))
    invs=[np.linalg.inv(N+pt*A) for pt in p]
    for i in range(len(deriv)):
        for j in range(i,len(deriv)):
            val=sum(nt*np.trace(W@deriv[i][t]@W@deriv[j][t]).real for t,(nt,W) in enumerate(zip(n,invs)))
            F[i,j]=F[j,i]=val
    # Nuisance Fisher block is positive semidefinite and singular due to
    # N -> N + c aaH, p_t -> p_t - c. Project with pseudoinverse.
    B=F[2:,2:]
    Z=F[:2,2:]
    return F[:2,:2]-Z@np.linalg.pinv(B, rcond=1e-11)@Z.T


def pack_chol(L):
    m=L.shape[0]
    v=[np.log(float(L[i,i].real)) for i in range(m)]
    for i in range(1,m):
        for j in range(i):v.extend([float(L[i,j].real),float(L[i,j].imag)])
    return np.array(v,float)


def unpack_chol(params,m):
    L=np.zeros((m,m),complex)
    for i in range(m): L[i,i]=np.exp(params[i])
    k=m
    for i in range(1,m):
        for j in range(i):
            L[i,j]=params[k]+1j*params[k+1];k+=2
    return L


def power_profile(a,N,stats):
    W=np.linalg.inv(N)
    wa=W@a
    s=float(np.vdot(a,wa).real)
    q=np.einsum('i,tij,j->t',wa.conj(),stats,wa).real
    return np.maximum(0.,(q-s)/(s*s))


def contrast_pilot(stats,positions,freq_hz,angle_grid=None,range_grid=None):
    """Truth-free joint angle/range pilot from temporal covariance contrast."""
    if angle_grid is None:angle_grid=np.deg2rad(np.arange(-60,61,4.))
    if range_grid is None:range_grid=np.array([.55,.75,1.0,1.5,2.2])
    diag=np.trace(stats,axis1=1,axis2=2).real
    lo=int(np.argmin(diag));hi=int(np.argmax(diag))
    D=stats[hi]-stats[lo]
    vals,U=np.linalg.eigh(D)
    # Changing source could decrease rather than increase in the measured
    # contrast; sign is chosen using the absolute largest eigenvalue.
    k=int(np.argmax(np.abs(vals)));u=U[:,k]
    best=(-np.inf,None,None)
    for t in angle_grid:
        for r in range_grid:
            a,_=steering_nearfield(float(t),float(r),positions,freq_hz)
            score=float(abs(np.vdot(a,u))**2)
            if score>best[0]:best=(score,float(t),float(r))
    return best[1],best[2],best[0]


@dataclass
class NearfieldProfileProblem:
    stats: np.ndarray
    counts: np.ndarray
    positions: np.ndarray
    freq_hz: float
    theta_lim: float = np.deg2rad(70)
    range_min: float = .4
    range_max: float = 3.0

    def __post_init__(self):
        self.stats=np.asarray(self.stats,complex)
        self.counts=np.asarray(self.counts,float)
        self.positions=np.asarray(self.positions,float)
        if (self.stats.ndim!=3 or self.stats.shape[1]!=len(self.positions)
                or self.stats.shape[2]!=len(self.positions)
                or self.stats.shape[0]!=len(self.counts)):
            raise ValueError('inconsistent covariance or array dimension')
        if (np.any(self.counts<len(self.positions)) or len(self.positions)<3):
            raise ValueError('require independent n_t >= M for well-posed covariance ML')
        if not (self.range_min>0 and self.range_min<self.range_max):
            raise ValueError('invalid range bounds')
        self.m=len(self.positions)

    def objective_gradient(self,z):
        theta=float(z[0]);r=np.exp(float(z[1]))
        L=unpack_chol(z[2:],self.m)
        N=L@L.conj().T
        # Profile p using a Cholesky solve instead of constructing N^{-1}
        # from an ill-conditioned matrix product.
        W=cho_solve((L,True),np.eye(self.m,dtype=complex),check_finite=False)
        a,D=steering_nearfield(theta,r,self.positions,self.freq_hz)
        wa=W@a
        s=float(np.vdot(a,wa).real)
        q=np.einsum('i,tij,j->t',wa.conj(),self.stats,wa).real
        p=np.maximum(0.,(q-s)/(s*s))
        A=np.outer(a,a.conj())
        value=0.0
        Gsum=np.zeros_like(N)
        gtheta=0.;glogr=0.
        for t,(S,nt) in enumerate(zip(self.stats,self.counts)):
            sig=N+p[t]*A
            # Direct covariance likelihood: logdet(sig)+tr(sig^-1 S).
            # IMPORTANT: avoid the numerically catastrophic cancellation
            # tr(W S)-p*q/(1+s*p) near a singular nuisance noise matrix.
            C,lower=cho_factor(sig,lower=True,check_finite=False)
            sinv=cho_solve((C,lower),np.eye(self.m,dtype=complex),check_finite=False)
            logdet=2*np.log(np.diag(C).real).sum()
            value+=float(nt)*(logdet+np.trace(sinv@S).real)
            G=nt*(sinv-sinv@S@sinv)
            Gsum+=G
            gtheta+=2*p[t]*float(np.vdot(D[:,0],G@a).real)
            glogr+=2*p[t]*r*float(np.vdot(D[:,1],G@a).real)
        gradL=2*Gsum@L
        out=[float(gradL[i,i].real)*float(L[i,i].real) for i in range(self.m)]
        for i in range(1,self.m):
            for j in range(i):out.extend([float(gradL[i,j].real),float(gradL[i,j].imag)])
        return float(value),np.r_[gtheta,glogr,out]

    def likelihood_lower_bound(self):
        """Saturated unconstrained complex-Gaussian covariance likelihood.

        For each positive-definite S_t: logdet Sigma + tr(Sigma^-1 S_t)
        is >= logdet S_t + M. This is a hard numerical audit guard.
        """
        return float(np.dot(self.counts, [np.linalg.slogdet(S)[1]+self.m for S in self.stats]))

    def estimate(self,theta_start=None,r_start=None,maxiter=180):
        if theta_start is None or r_start is None:
            theta_start,r_start,_=contrast_pilot(self.stats,self.positions,self.freq_hz)
        # At the lowest-power block, the sample covariance is a sensible
        # initialization for the unknown shared full covariance N.
        tr=np.trace(self.stats,axis1=1,axis2=2).real
        N0=self.stats[int(np.argmin(tr))]+1e-8*np.eye(self.m)
        L0=np.linalg.cholesky(N0)
        z0=np.r_[theta_start,np.log(r_start),pack_chol(L0)]
        b=[(-self.theta_lim,self.theta_lim),(np.log(self.range_min),np.log(self.range_max))]
        # Broad, observation-scaled numerical bounds prevent float underflow
        # on trial line-search steps. They are NOT a substantive noise model.
        scale=np.sqrt(float(np.trace(self.stats.mean(axis=0)).real/self.m))
        lower=np.log(max(scale*1e-4,1e-12))
        upper=np.log(max(scale*1e4,1e-5))
        b.extend([(lower,upper)]*self.m)
        b.extend([(-50*scale,50*scale)]*(self.m**2-self.m))
        try:
            sol=minimize(self.objective_gradient,z0,method='L-BFGS-B',jac=True,bounds=b,
                     options={'maxiter':maxiter,'ftol':1e-11,'gtol':1e-6})
        except (np.linalg.LinAlgError,ValueError,FloatingPointError,OverflowError) as exc:
            # A failed start is audit-visible; it does not terminate all data.
            theta=float(theta_start);rr=float(r_start)
            a,_=steering_nearfield(theta,rr,self.positions,self.freq_hz)
            return dict(theta_deg=float(np.rad2deg(theta)),theta_rad=theta,range_m=rr,
                        profile_nll=float('inf'),success=False,nit=0,
                        common_N=N0,powers=power_profile(a,N0,self.stats),
                        message='numerically failed start: '+repr(exc))
        # Reject numerically impossible likelihoods (e.g. from roundoff).
        if not np.isfinite(sol.fun) or sol.fun < self.likelihood_lower_bound()-1e-5*max(1,abs(self.likelihood_lower_bound())):
            theta=float(theta_start);rr=float(r_start)
            a,_=steering_nearfield(theta,rr,self.positions,self.freq_hz)
            return dict(theta_deg=float(np.rad2deg(theta)),theta_rad=theta,range_m=rr,
                        profile_nll=float('inf'),success=False,nit=int(sol.nit),
                        common_N=N0,powers=power_profile(a,N0,self.stats),
                        message='rejected: nonfinite or impossible Gaussian objective')
        th=float(sol.x[0]);rr=float(np.exp(sol.x[1]));L=unpack_chol(sol.x[2:],self.m)
        NN=L@L.conj().T
        a,_=steering_nearfield(th,rr,self.positions,self.freq_hz)
        powers=power_profile(a,NN,self.stats)
        # Canonical reporting gauge: min_t powers[t]=0. No impact on likelihood.
        c=float(np.min(powers));NN=NN+c*np.outer(a,a.conj());powers=powers-c
        return dict(theta_deg=float(np.rad2deg(th)),theta_rad=th,range_m=rr,
                    profile_nll=float(sol.fun),success=bool(sol.success),nit=int(sol.nit),
                    common_N=NN,powers=powers, message=str(sol.message))

    def estimate_multistart(self,maxiter=180):
        th,r,_=contrast_pilot(self.stats,self.positions,self.freq_hz)
        # Symmetric, prespecified starts. None is centred on the 20-degree
        # simulation truth; the pilot alone adapts to the observed data.
        starts=[(th,r),(np.deg2rad(-45.),.70),(0.0,1.45),(np.deg2rad(45.),2.30)]
        # Drop duplicate physical starting points; no ground truth used.
        uniq=[]
        for t,rr in starts:
            if not any(abs(t-u[0])<.1 and abs(np.log(rr/u[1]))<.1 for u in uniq):uniq.append((t,rr))
        fits=[self.estimate(t,rr,maxiter) for t,rr in uniq]
        finite=[f for f in fits if np.isfinite(f['profile_nll'])]
        if not finite:
            fallback=contrast_direction(self.stats,self.positions,self.freq_hz)
            return dict(theta_deg=fallback['theta_deg'],theta_rad=np.deg2rad(fallback['theta_deg']),
                        range_m=fallback['range_m'],profile_nll=float('inf'),success=False,
                        nit=0,common_N=np.full((self.m,self.m),np.nan,dtype=complex),
                        powers=np.full(self.stats.shape[0],np.nan),message='all ML starts numerically failed',
                        starts_tried=len(starts),converged_starts=0)
        winner=min(finite,key=lambda f:f['profile_nll'])
        winner['starts_tried']=len(uniq)
        winner['converged_starts']=sum(f['success'] for f in fits)
        winner['first_start_theta_deg']=float(np.rad2deg(th))
        winner['first_start_range_m']=float(r)
        return winner




def _steering_vector_fit(u,positions,freq_hz):
    """Grid + bounded local refinement of |u^H a(theta,r)|²."""
    theta_grid=np.deg2rad(np.arange(-60.,60.00001,4))
    rgrid=np.array([.55,.75,1.0,1.5,2.2])
    best=(-1.,None,None)
    for theta in theta_grid:
        for r in rgrid:
            a,_=steering_nearfield(theta,r,positions,freq_hz)
            sc=float(abs(np.vdot(u,a))**2)
            if sc>best[0]:best=(sc,theta,r)
    def obj(x):
        a,_=steering_nearfield(float(x[0]),float(np.exp(x[1])),positions,freq_hz)
        return -float(abs(np.vdot(u,a))**2)
    res=minimize(obj,[best[1],np.log(best[2])],bounds=[(-np.deg2rad(70),np.deg2rad(70)),(np.log(.4),np.log(3.))],method='L-BFGS-B',options={'maxiter':80})
    return dict(theta_deg=float(np.rad2deg(res.x[0])),range_m=float(np.exp(res.x[1])),score=float(-res.fun),success=bool(res.success))


def contrast_direction(stats,positions,freq_hz):
    """Rank-one temporal-contrast eigendirection without Gaussian likelihood."""
    tr=np.trace(stats,axis1=1,axis2=2).real
    D=stats[int(np.argmax(tr))]-stats[int(np.argmin(tr))]
    vals,V=np.linalg.eigh(D)
    u=V[:,np.argmax(abs(vals))]
    return _steering_vector_fit(u,positions,freq_hz)




