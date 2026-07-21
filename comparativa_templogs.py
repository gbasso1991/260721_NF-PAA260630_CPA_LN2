#%% Librerias y paquetes 
import numpy as np
from uncertainties import ufloat, unumpy
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import chardet
import re
from datetime import datetime
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
pendiente_HvsI = 3716.3 # 1/m
ordenada_HvsI = 1297.0 # A/m
# from clase_resultados import ResultadosESAR
#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt).
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python')
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']])

    time = np.array([(t-timestamp[0]).total_seconds() for t in timestamp])
    return timestamp,time,temp_CH1, temp_CH2
#%% Curvatura
def curvatura(T):
    dT=savgol_filter(T,window_length=11,polyorder=3,deriv=1,delta=1.0)
    dTT=savgol_filter(T,window_length=11,polyorder=3,deriv=2,delta=1.0)
    curv=np.abs(dTT)/(1+dT**2)**1.5
    return dT,dTT,curv
#%% Exponencial
def expo(t,A,B,tau):
    return A + B*np.exp(-t/tau)
def biexpo(t,A,B1,tau1,B2,tau2):
    return A + B1*np.exp(-t/tau1) + B2*np.exp(-t/tau2)

#%% 1 - 500 uL EG 51 FF 49 LN2 RF300 Idc [100, 090 , 080 , 070, 060, 050]
print('-'*50,'\nEG 51 FF 49 LN2 RF300- Idc= [100, 090 , 080 , 070, 060, 050] dA','\n')
temps_500_EG51_FF49 = glob("data/*.csv",recursive=True)
temps_500_EG51_FF49.sort()
for p in temps_500_EG51_FF49:
    print('  -',os.path.basename(p))
print('\n')
#%%
Idc_values = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
H0=[(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in Idc_values]  
t,t_min=[],[]
T,T_min=[],[]
Indx_min,dT=[],[]


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,9),constrained_layout=True)
ax1.set_title('1.1 - EG 51% FF 49% - LN2 --> RF - Idc = [100, 090, 080] dA',loc='left')
ax2.set_title('1.2 - EG 51% FF 49% - LN2 --> RF - Idc = [070, 060 , 050] dA',loc='left')

for i,p in enumerate(temps_500_EG51_FF49[:3]):
    _,time,temp_CH1, _ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax1.plot(time,temp_CH1,label=f'{H0[i]:.1f} kA/m')
    #ax1.plot(time,(np.gradient(temp_CH1,time)))

for i,p in enumerate(temps_500_EG51_FF49[3:]):
    _,time,temp_CH1, _ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax2.plot(time,temp_CH1,label=f'{H0[i+3]:.1f} kA/m')
    #ax2.plot(time,(np.gradient(temp_CH1,time)))

for a in (ax1,ax2):
    a.grid()
    a.legend(title='f = 300 kHz',loc='lower right',shadow=True,frameon=True)
    a.set_ylabel('T (°C)')
ax1.set_xlim(0,140)
ax2.set_xlim(0,600)
ax2.set_xlabel('t (s)')

#%% Curvatura
fig25,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(temps_500_EG51_FF49):
    _,time,temp,_ = lector_templog(p)
    indx_min = np.argmin(temp)
    indx_max = indx_min + np.argmax(temp[indx_min:])
    print('\n',os.path.basename(p))
    print(f'T min = {temp[indx_min]:.1f} C ({temp[indx_min]+273:.1f} K) alcanzada en {time[indx_min]:.1f} s')
    print(f'T max = {temp[indx_max]:.1f} C ({temp[indx_max]+273:.1f} K) alcanzada en {time[indx_max]:.1f} s')

    t_curv = time[indx_min:indx_max]
    T_curv = temp[indx_min:indx_max]
    # fig,ax=plt.subplots(figsize=(8,4),constrained_layout=True)

    # ax.plot(time,temp)
    # ax.plot(t_curv,T_curv,'o')
    # plt.show()

    mask = (T_curv > -158) & (T_curv < 0)

    T_curv = T_curv[mask]
    t_curv = t_curv[mask]

    _,_,curv = curvatura(T_curv)

    ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
    ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')
    ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f}')

ax.axhspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)

ax2.axvspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-75,-50,color='tab:purple',alpha=0.2,zorder=-1)

ax.set_xlim(0,300)
ax.set_ylim(-160,0)
ax2.set_xlim(-160,-100)
ax3.set_xlim(-100,0)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(title='H$_0$ (kA/m)',ncol=2,shadow=True,frameon=True)

plt.suptitle('EG 51% FF 49%\nLN2 --> RF\nIdc = [100, 090, 080, 070, 060, 050]  dA')
#%% Ajuste exponencial y bi-exponencial
# Taux=[]
# fits_exp=[]
# fits_biexp=[]
# residuos_exp=[]
# residuos_biexp=[]

# for i,(x,y) in enumerate(zip(t,T)):

#     t_aux,T_aux = x[Indx_min[i]:],y[Indx_min[i]:]

#     # Ajuste exponencial simple
#     p0_exp = [T_aux[-1],T_aux[0]-T_aux[-1],200]

#     (A,B,tau),_ = curve_fit(
#         expo,
#         t_aux,
#         T_aux,
#         p0=p0_exp
#     )

#     Tfit_exp = expo(t_aux,A,B,tau)
#     res_exp = T_aux - Tfit_exp

#     # Ajuste bi-exponencial
#     p0_bi = [
#         T_aux[-1],
#         0.7*(T_aux[0]-T_aux[-1]),50,
#         0.3*(T_aux[0]-T_aux[-1]),300
#     ]

#     (A2,B1,tau1,B2,tau2),_ = curve_fit(
#         biexpo,
#         t_aux,
#         T_aux,
#         p0=p0_bi,
#         maxfev=10000
#     )

#     Tfit_bi = biexpo(t_aux,A2,B1,tau1,B2,tau2)
#     res_bi = T_aux - Tfit_bi

#     # -------------------------------------------------------
#     # Métricas
#     # -------------------------------------------------------
#     sigma = 0.1
#     N = len(T_aux)

#     k_exp = 3
#     k_bi  = 5

#     rss_exp = np.sum(res_exp**2)
#     rss_bi  = np.sum(res_bi**2)

#     r2_exp = 1 - rss_exp/np.sum((T_aux-T_aux.mean())**2)
#     r2_bi  = 1 - rss_bi/np.sum((T_aux-T_aux.mean())**2)

#     rmse_exp = np.sqrt(np.mean(res_exp**2))
#     rmse_bi  = np.sqrt(np.mean(res_bi**2))

#     chi2_exp = np.sum((res_exp/sigma)**2)
#     chi2_bi  = np.sum((res_bi/sigma)**2)

#     chi2red_exp = chi2_exp/(N-k_exp)
#     chi2red_bi  = chi2_bi/(N-k_bi)

#     aic_exp = N*np.log(rss_exp/N) + 2*k_exp
#     aic_bi  = N*np.log(rss_bi/N) + 2*k_bi

#     delta_aic = aic_exp - aic_bi

#     print(f'H0 = {H0[i]:.1f} kA/m')
#     print(f'Exp:   tau = {tau:6.1f} s | RMSE = {rmse_exp:6.3f} °C | R² = {r2_exp:7.5f} | χ²ν = {chi2red_exp:8.2f} | AIC = {aic_exp:8.1f}')
#     print(f'BiExp: tau1 = {tau1:6.1f} s | tau2 = {tau2:6.1f} s | RMSE = {rmse_bi:6.3f} °C | R² = {r2_bi:7.5f} | χ²ν = {chi2red_bi:8.2f} | AIC = {aic_bi:8.1f}')
#     print(f'ΔAIC = {delta_aic:.1f}')
#     print('-'*100)

#     Taux.append(T_aux)
#     fits_exp.append(Tfit_exp)
#     fits_biexp.append(Tfit_bi)
#     residuos_exp.append(res_exp)
#     residuos_biexp.append(res_bi)

#     fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),constrained_layout=True)
#     ax1.plot(t_aux-t_aux[0],T_aux,'.',ms=3,label='Datos')

#     ax1.plot(t_aux-t_aux[0],Tfit_exp,'-',lw=2,label='Exp.')

#     ax1.plot(t_aux-t_aux[0],Tfit_bi,'--',lw=2,label='Bi-exp.')

#     texto = (rf'$\chi^2_{{\nu,\mathrm{{exp}}}}={chi2red_exp:.2f}$' '\n'
#         rf'$\chi^2_{{\nu,\mathrm{{bi}}}}={chi2red_bi:.2f}$' '\n'
#         rf'$\Delta AIC={delta_aic:.1f}$')

#     ax1.text(0.8,0.5,texto,transform=ax1.transAxes,ha='left',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))

#     ax1.set_title('Temperatura vs tiempo',loc='left')
#     ax1.set_ylabel('T (°C)')
#     ax1.set_xlabel('t (s)')

#     ax2.set_title('Residuos vs temperatura',loc='left')

#     ax2.plot(T_aux,res_exp,'.-',label='Residuo exp.')

#     ax2.plot(T_aux,res_bi,'.-',label='Residuo bi-exp.')

#     ax2.axhline(0,color='k',lw=1)

#     ax2.set_xlim(-175,0)
#     ax2.set_xlabel('T (°C)')
#     ax2.set_ylabel('Residuo (°C)')

#     for a in [ax1,ax2]:
#         a.grid()
#         a.legend()

#     plt.suptitle(f'2.5 - EG 55% FF 45% - LN2 → RF - H$_0$ = {H0[i]:.1f} kA/m')
# #%% Ploteo todos los resuiduos
# fig26,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
# ax.set_title('Residuos vs Temp - EG 55% FF 45%',loc='left')

# for i in range(len(residuos_biexp)):
#     plt.plot(Taux[i], residuos_biexp[i],'.-',label=f'{H0[i]:.1f}')
# ax.axhline(0,color='k',lw=1)
# ax.axvspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
# ax.axvspan(-60,-40,color='tab:orange',alpha=0.2,zorder=-1)
# ax.grid()
# ax.legend(title=' H$_0$ (kA/m)',loc='lower right',ncol=1)
# ax.set_xlim(-170,0)
# ax.set_ylim(-10,10)
#%% salvo figuras
fig.savefig('0_EG51_FF49_LN2_to_RF_150_125_100_075_050_000.png',dpi=300)
fig25.savefig('1_EG51_FF49_templogs_curvatura.png',dpi=300)

# fig2.savefig('2_EG55_FF45_LN2_to_RF_150_125_100.png',dpi=300)
# fig23.savefig('2_EG55_FF45_grad_temp_150_125_100.png',dpi=300)
# fig24.savefig('2_EG55_FF45_grad_temp_075_050_000.png',dpi=300)
# fig25.savefig('2_EG55_FF45_templogs_curvatura.png',dpi=300)
# fig26.savefig('2_EG55_FF45_templogs_residuos.png',dpi=300)

#%%3 - 500 uL EG 53 FF 47 LN2 RF 300 Idc = [150, 125, 100, 075, 050]
print('-'*50,'\nEG 53 FF 47 LN2 RF300- Idc= [150, 125, 100, 075, 050] dA','\n')

temps_500_EG53_FF47 = glob("3_EG53_FF47_LN2_to_RF_150_125_100_075_050/*.csv",recursive=True)
temps_500_EG53_FF47.sort()
for p in temps_500_EG53_FF47:
    print('  -',os.path.basename(p))
Idc_values = [15.0, 12.5, 10.0, 7.5, 5.0,0]
H0=[(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in Idc_values]  
t,t_min=[],[]
T,T_min=[],[]
Indx_min,dT=[],[]

fig3,(ax1,ax2) = plt.subplots(2,1,figsize=(9,9),constrained_layout=True)
ax1.set_title('3.1 - EG 53% FF 47% - LN2 --> RF - Idc = [150, 125, 100] dA',loc='left')
ax2.set_title('3.2 - EG 53% FF 47% - LN2 --> RF - Idc = [75, 50, 00] dA',loc='left')

for i,p in enumerate(temps_500_EG53_FF47[:3]):
    _,time,temp_CH1, _ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax1.plot(time,temp_CH1,label=f'{H0[i]:.1f} kA/m')
    ax1.plot(time,(np.gradient(temp_CH1,time)))

for i,p in enumerate(temps_500_EG53_FF47[3:]):
    _,time,temp_CH1, _ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax2.plot(time,temp_CH1,label=f'{H0[i+3]:.1f} kA/m' if i!=2 else '0 kA/m')
    ax2.plot(time,(np.gradient(temp_CH1,time)))

for a in (ax1,ax2):
    a.set_xlim(0,175)
    a.grid()
    a.legend(title='f = 300 kHz',loc='lower right',shadow=True,frameon=True)
    a.set_ylabel('T (°C)')
ax2.set_xlim(60,350)
ax2.set_xlabel('t (s)')

# Gradiente
col=['C0','C1','C2']
fig33,axs=plt.subplots(3,1,figsize=(10,5.5),constrained_layout=True,sharex=True)

for i,(x,y,z) in enumerate(zip(t[:3],T[:3],dT[:3])):
    l1 = axs[i].plot(x[Indx_min[i]:],z[Indx_min[i]:],'.-',c=col[i],label='dT/dt')
    ax2 = axs[i].twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('T (K)')
    Tplot = y[Indx_min[i]:] + 273
    l2 = ax2.plot(x[Indx_min[i]:],Tplot,c=col[i],label='T')
    ax2.set_yticks([Tplot.min(),Tplot.max()])
    ax2.set_yticklabels([f'{Tplot.min():.0f}',f'{Tplot.max():.0f}'])
    ax2.minorticks_off()

    handles = l1 + l2
    labels = [h.get_label() for h in handles]

    axs[i].legend(handles, labels, frameon=True, shadow=True,title=f'H$_0$ = {H0[i]:.1f} kA/m',loc='lower right',ncol=2)

for a in axs:
    a.set_ylabel('dT/dt (°C/s)')
    a.set_xlim(min(t_min[:3])[0]-1,)
    a.grid()
    a.set_ylim(-1,3.5)
axs[2].set_xlabel('t (s)')
fig33.suptitle('3.3 - EG 53% FF 47% - LN2 --> RF - Idc = [150, 125, 100] dA')

fig34,axs=plt.subplots(3,1,figsize=(10,5.5),constrained_layout=True,sharex=True)

for i,(x,y,z) in enumerate(zip(t[3:],T[3:],dT[3:])):
    l1 = axs[i].plot(x[Indx_min[i]:],z[Indx_min[i]:],'.-',c=col[i],label='dT/dt')
    ax2 = axs[i].twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('T (K)')
    Tplot = y[Indx_min[i]:] + 273
    l2 = ax2.plot(x[Indx_min[i]:],Tplot,c=col[i],label='T')
    ax2.set_yticks([Tplot.min(),273])
    ax2.set_yticklabels([f'{Tplot.min():.0f}','273'])
    ax2.minorticks_off()
    ax2.set_ylim(98,273)

    handles = l1 + l2
    labels = [h.get_label() for h in handles]

    axs[i].legend(handles, labels, frameon=True, shadow=True,title=f'H$_0$ = {H0[i]:.1f} kA/m',loc='lower right',ncol=2)

for a in axs:
    a.set_ylabel('dT/dt (°C/s)')
    a.set_xlim(min(np.concatenate(t_min)[3:])-2,250)
    a.set_yticks([-1,-0.5,0,0.5,1,1.5,2,2.5])
    # a.set_yticklabels(['-1','-0.5','0','0.5','1','1.5','2','2.5'])    
    a.set_ylim(-1,2.5)
    
    a.grid()
axs[2].set_xlabel('t (s)')
fig34.suptitle('3.4 - EG 53% FF 47% - LN2 --> RF - Idc = [075, 050, 100] dA')

fig35,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(temps_500_EG53_FF47):
    _,time,temp, _ = lector_templog(p)
    indx_min=np.nonzero(temp==min(temp))[0][0]
    indx_max=np.nonzero(temp==max(temp[indx_min:]))[0][0]
    print(temp[indx_min],'-',temp[indx_max])
    t_curv = time[indx_min:indx_max]
    T_curv = temp[indx_min:indx_max]
    mask = (T_curv > -155) & (T_curv < 0)
    T_curv = T_curv[mask]
    t_curv = t_curv[mask]
    _,_,curv = curvatura(T_curv)
    
    ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.0f} kA/m')
    ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f} kA/m')
    ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f} kA/m')
ax.axhspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-70,-40,color='tab:orange',alpha=0.2,zorder=-1)

ax2.axvspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-70,-40,color='tab:orange',alpha=0.2,zorder=-1)

ax.set_xlim(0,250)
ax2.set_xlim(-160,-100)
ax3.set_xlim(-100,00)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')
for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(ncol=2)
plt.suptitle('EG 53% FF 47%\nLN2 --> RF\nIdc = [150, 125, 100, 075, 050, 000] dA')    
#%%% Ajuste exponencial y bi-exponencial
Taux=[]
fits_exp=[]
fits_biexp=[]
residuos_exp=[]
residuos_biexp=[]

for i,(x,y) in enumerate(zip(t,T)):

    t_aux,T_aux = x[Indx_min[i]:],y[Indx_min[i]:]

    # Ajuste exponencial simple
    p0_exp = [T_aux[-1],T_aux[0]-T_aux[-1],200]

    (A,B,tau),_ = curve_fit(
        expo,
        t_aux,
        T_aux,
        p0=p0_exp
    )

    Tfit_exp = expo(t_aux,A,B,tau)
    res_exp = T_aux - Tfit_exp

    # Ajuste bi-exponencial
    p0_bi = [
        T_aux[-1],
        0.7*(T_aux[0]-T_aux[-1]),50,
        0.3*(T_aux[0]-T_aux[-1]),300
    ]

    (A2,B1,tau1,B2,tau2),_ = curve_fit(
        biexpo,
        t_aux,
        T_aux,
        p0=p0_bi,
        maxfev=10000
    )

    Tfit_bi = biexpo(t_aux,A2,B1,tau1,B2,tau2)
    res_bi = T_aux - Tfit_bi

    # -------------------------------------------------------
    # Métricas
    # -------------------------------------------------------
    sigma = 0.1
    N = len(T_aux)

    k_exp = 3
    k_bi  = 5

    rss_exp = np.sum(res_exp**2)
    rss_bi  = np.sum(res_bi**2)

    r2_exp = 1 - rss_exp/np.sum((T_aux-T_aux.mean())**2)
    r2_bi  = 1 - rss_bi/np.sum((T_aux-T_aux.mean())**2)

    rmse_exp = np.sqrt(np.mean(res_exp**2))
    rmse_bi  = np.sqrt(np.mean(res_bi**2))

    chi2_exp = np.sum((res_exp/sigma)**2)
    chi2_bi  = np.sum((res_bi/sigma)**2)

    chi2red_exp = chi2_exp/(N-k_exp)
    chi2red_bi  = chi2_bi/(N-k_bi)

    aic_exp = N*np.log(rss_exp/N) + 2*k_exp
    aic_bi  = N*np.log(rss_bi/N) + 2*k_bi

    delta_aic = aic_exp - aic_bi

    print(f'H0 = {H0[i]:.1f} kA/m')
    print(f'Exp:   tau = {tau:6.1f} s | RMSE = {rmse_exp:6.3f} °C | R² = {r2_exp:7.5f} | χ²ν = {chi2red_exp:8.2f} | AIC = {aic_exp:8.1f}')
    print(f'BiExp: tau1 = {tau1:6.1f} s | tau2 = {tau2:6.1f} s | RMSE = {rmse_bi:6.3f} °C | R² = {r2_bi:7.5f} | χ²ν = {chi2red_bi:8.2f} | AIC = {aic_bi:8.1f}')
    print(f'ΔAIC = {delta_aic:.1f}')
    print('-'*100)

    Taux.append(T_aux)
    fits_exp.append(Tfit_exp)
    fits_biexp.append(Tfit_bi)
    residuos_exp.append(res_exp)
    residuos_biexp.append(res_bi)

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),constrained_layout=True)
    ax1.plot(t_aux-t_aux[0],T_aux,'.',ms=3,label='Datos')

    ax1.plot(t_aux-t_aux[0],Tfit_exp,'-',lw=2,label='Exp.')

    ax1.plot(t_aux-t_aux[0],Tfit_bi,'--',lw=2,label='Bi-exp.')

    texto = (rf'$\chi^2_{{\nu,\mathrm{{exp}}}}={chi2red_exp:.2f}$' '\n'
        rf'$\chi^2_{{\nu,\mathrm{{bi}}}}={chi2red_bi:.2f}$' '\n'
        rf'$\Delta AIC={delta_aic:.1f}$')

    ax1.text(0.8,0.5,texto,transform=ax1.transAxes,ha='left',va='top',bbox=dict(boxstyle='round',facecolor='white',alpha=0.85))

    ax1.set_title('Temperatura vs tiempo',loc='left')
    ax1.set_ylabel('T (°C)')
    ax1.set_xlabel('t (s)')

    ax2.set_title('Residuos vs temperatura',loc='left')

    ax2.plot(T_aux,res_exp,'.-',label='Residuo exp.')

    ax2.plot(T_aux,res_bi,'.-',label='Residuo bi-exp.')

    ax2.axhline(0,color='k',lw=1)

    ax2.set_xlim(-175,0)
    ax2.set_xlabel('T (°C)')
    ax2.set_ylabel('Residuo (°C)')

    for a in [ax1,ax2]:
        a.grid()
        a.legend()

    plt.suptitle(f'2.5 - EG 55% FF 45% - LN2 → RF - H$_0$ = {H0[i]:.1f} kA/m')
#%% Ploteo todos los resuiduos
fig36,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
ax.set_title('Residuos vs Temp - EG 53% FF 47%',loc='left')

for i in range(len(residuos_biexp)):
    plt.plot(Taux[i], residuos_biexp[i],'.-',label=f'{H0[i]:.1f}')
ax.axhline(0,color='k',lw=1)
ax.axvspan(-150,-130,color='tab:red',alpha=0.2,zorder=-1)
ax.axvspan(-60,-40,color='tab:orange',alpha=0.2,zorder=-1)
ax.grid()
ax.legend(title=' H$_0$ (kA/m)',loc='lower right',ncol=1)
ax.set_xlim(-170,0)
ax.set_ylim(-10,10)

#%%salvo figuras
fig3.savefig('3_EG53FF47_LN2_RF_150_125_100.png',dpi=300)
fig33.savefig('3_EG53FF47_grad_temp_150_125_100.png',dpi=300)
fig34.savefig('3_EG53FF47_grad_temp_075_050_000.png',dpi=300)
fig35.savefig('3_EG53FF47_templogs_curvatura.png',dpi=300)
fig36.savefig('3_EG53FF47_templogs_residuos.png',dpi=300)
#%%4 - 500 uL EG 51 FF 49 LN2 RF 300 - Idc= [150, 125, 100, 075, 050]
print('-'*50,'\nEG 51 FF 49 LN2 RF300 - Idc= [150, 125, 100, 075, 050] dA','\n')
temps_500_EG51_FF49 = glob("4_EG51_FF49_LN2_to_RF_150_125_100_075_050/*.csv",recursive=True)
temps_500_EG51_FF49.sort()
for p in temps_500_EG51_FF49:
    print('  -',os.path.basename(p))
Idc_values = [15.0, 12.5, 10.0, 7.5, 5.0,0]
H0=[(h*pendiente_HvsI+ordenada_HvsI)/1000 for h in Idc_values]
t,t_min=[],[]
T,T_min=[],[]
Indx_min,dT=[],[]

fig4,(ax1,ax2) = plt.subplots(2,1,figsize=(9,9),constrained_layout=True)
ax1.set_title('4.1 - EG 51% FF 49% - LN2 --> RF - Idc = [150, 125, 100] dA',loc='left')
ax2.set_title('4.2 - EG 51% FF 49% - LN2 --> RF - Idc = [75, 50, 00] dA',loc='left')

for i,p in enumerate(temps_500_EG51_FF49[:3]):
    _,time,temp_CH1,_ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax1.plot(time,temp_CH1,label=f'{H0[i]:.1f} kA/m')
    ax1.plot(time,(np.gradient(temp_CH1,time)))

for i,p in enumerate(temps_500_EG51_FF49[3:]):
    _,time,temp_CH1,_ = lector_templog(p)
    t.append(time)
    T.append(temp_CH1)
    dT.append(np.gradient(temp_CH1,time))
    indx_min=np.nonzero(temp_CH1==min(temp_CH1))[0]
    t_min.append(time[indx_min])
    T_min.append(temp_CH1[indx_min])
    Indx_min.append(indx_min[0])
    print(f'Temp minima = {temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} C ({temp_CH1[np.nonzero(temp_CH1==min(temp_CH1))][0]+273:.1f} K) alcanzada en {time[np.nonzero(temp_CH1==min(temp_CH1))][0]:.1f} s')
    ax2.plot(time,temp_CH1,label=f'{H0[i+3]:.1f} kA/m' if i!=2 else '0 kA/m')
    ax2.plot(time,(np.gradient(temp_CH1,time)))

ax1.set_xlim(0,160)
ax2.set_xlim(0,350)

for a in (ax1,ax2):
    a.grid()
    a.legend(title='f = 300 kHz',loc='lower right',shadow=True,frameon=True)
    a.set_ylabel('T (°C)')
ax2.set_xlabel('t (s)')

#% Gradiente
col=['C0','C1','C2']
fig43,axs=plt.subplots(3,1,figsize=(10,5.5),constrained_layout=True,sharex=True)

for i,(x,y,z) in enumerate(zip(t[:3],T[:3],dT[:3])):
    l1 = axs[i].plot(x[Indx_min[i]:],z[Indx_min[i]:],'.-',c=col[i],label='dT/dt')
    ax2 = axs[i].twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('T (K)')
    Tplot = y[Indx_min[i]:] + 273
    l2 = ax2.plot(x[Indx_min[i]:],Tplot,c=col[i],label='T')
    ax2.set_yticks([Tplot.min(),Tplot.max()])
    ax2.set_yticklabels([f'{Tplot.min():.0f}',f'{Tplot.max():.0f}'])
    ax2.minorticks_off()

    handles = l1 + l2
    labels = [h.get_label() for h in handles]

    axs[i].legend(handles, labels, frameon=True, shadow=True,title=f'H$_0$ = {H0[i]:.1f} kA/m',loc='lower right',ncol=2)

for a in axs:
    a.set_ylabel('dT/dt (°C/s)')
    a.set_xlim(min(t_min[:3])[0]-1,)
    a.grid()
    #a.set_ylim(-1,2.5)

axs[2].set_xlabel('t (s)')
fig43.suptitle('4.3 - EG 51% FF 49% - LN2 --> RF - Idc = [150, 125, 100] dA')

fig44,axs=plt.subplots(3,1,figsize=(10,5.5),constrained_layout=True,sharex=True)

for i,(x,y,z) in enumerate(zip(t[3:],T[3:],dT[3:])):
    l1 = axs[i].plot(x[Indx_min[i]:],z[Indx_min[i]:],'.-',c=col[i],label='dT/dt')
    ax2 = axs[i].twinx()
    ax2.set_yscale('log')
    ax2.set_ylabel('T (K)')
    Tplot = y[Indx_min[i]:] + 273
    l2 = ax2.plot(x[Indx_min[i]:],Tplot,c=col[i],label='T')
    ax2.set_yticks([Tplot.min(),273])
    ax2.set_yticklabels([f'{Tplot.min():.0f}','273'])
    ax2.minorticks_off()
    ax2.set_ylim(98,273)

    handles = l1 + l2
    labels = [h.get_label() for h in handles]

    axs[i].legend(handles, labels, frameon=True, shadow=True,title=f'H$_0$ = {H0[i]:.1f} kA/m',loc='lower right',ncol=2)

for a in axs:
    a.set_ylabel('dT/dt (°C/s)')
    a.set_xlim(min(np.concatenate(t_min)[3:-1]),250)
    a.set_yticks([-1,-0.5,0,0.5,1,1.5,2,2.5])
    #a.set_ylim(-1,2.5)
    a.grid()

axs[2].set_xlabel('t (s)')
fig44.suptitle('4.4 - EG 51% FF 49% - LN2 --> RF - Idc = [075, 050, 100] dA')
#%%
fig45,(ax,ax2,ax3) = plt.subplots(3,1,figsize=(10,10),constrained_layout=True)
ax.set_title('Temp vs time',loc='left')
ax2.set_title('Curvatura vs Temp',loc='left')
ax3.set_title('Curvatura vs Temp',loc='left')

for i,p in enumerate(temps_500_EG51_FF49[:-1]):
    _,time,temp, _ = lector_templog(p)
    indx_min=np.nonzero(temp==min(temp))[0][0]
    indx_max=np.nonzero(temp==max(temp[indx_min:]))[0][0]
    print(temp[indx_min],'-',temp[indx_max])
    t_curv = time[indx_min:indx_max]    
    T_curv = temp[indx_min:indx_max]
    mask= (T_curv > -160) & (T_curv <0)
    T_curv=T_curv[mask]
    t_curv=t_curv[mask]
    _,_,curv = curvatura(T_curv)
    ax.plot(t_curv-t_curv[0],T_curv,'.-',label=f'H$_0$ = {H0[i]:.0f} kA/m')
    ax2.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f} kA/m') 
    ax3.plot(T_curv,curv,'.-',label=f'H$_0$ = {H0[i]:.1f} kA/m')  
ax.axhspan(-155,-130,color='tab:red',alpha=0.2,zorder=-1)
ax.axhspan(-70,-40,color='tab:orange',alpha=0.2,zorder=-1)

ax2.axvspan(-155,-130,color='tab:red',alpha=0.2,zorder=-1)
ax3.axvspan(-70,-40,color='tab:orange',alpha=0.2,zorder=-1)

ax.set_xlim(0,250)
ax2.set_xlim(-160,-100)
ax3.set_xlim(-100,00)
ax.set_xlabel('t (s)')
ax.set_ylabel('T (°C)')
ax2.set_ylabel('Curvatura')
ax3.set_ylabel('Curvatura')
ax3.set_xlabel('T (°C)')
for a in [ax,ax2,ax3]:
    a.grid()
    a.legend(ncol=2)
plt.suptitle('EG 51% FF 49%\nLN2 --> RF\nIdc = [150, 125, 100, 075, 050, 000] dA')    

#%%% Ajuste exponencial y bi-exponencial
Taux=[]
fits_exp=[]
fits_biexp=[]
residuos_exp=[]
residuos_biexp=[]

for i,(x,y) in enumerate(zip(t[:-1],T[:-1])):

    t_aux,T_aux = x[Indx_min[i]:],y[Indx_min[i]:]

    # Ajuste exponencial simple
    p0_exp = [T_aux[-1],T_aux[0]-T_aux[-1],200]

    (A,B,tau),_ = curve_fit(expo,t_aux,T_aux,p0=p0_exp)

    Tfit_exp = expo(t_aux,A,B,tau)
    res_exp = T_aux - Tfit_exp

    # Ajuste bi-exponencial
    p0_bi = [T_aux[-1],
             0.7*(T_aux[0]-T_aux[-1]),50,
             0.3*(T_aux[0]-T_aux[-1]),300]

    (A2,B1,tau1,B2,tau2),_ = curve_fit(
        biexpo,
        t_aux,
        T_aux,
        p0=p0_bi,
        maxfev=10000
    )

    Tfit_bi = biexpo(t_aux,A2,B1,tau1,B2,tau2)
    res_bi = T_aux - Tfit_bi

    # Métricas
    sigma = 0.1
    N = len(T_aux)

    k_exp = 3
    k_bi  = 5

    rss_exp = np.sum(res_exp**2)
    rss_bi  = np.sum(res_bi**2)

    r2_exp = 1 - rss_exp/np.sum((T_aux-T_aux.mean())**2)
    r2_bi  = 1 - rss_bi/np.sum((T_aux-T_aux.mean())**2)

    rmse_exp = np.sqrt(np.mean(res_exp**2))
    rmse_bi  = np.sqrt(np.mean(res_bi**2))

    chi2_exp = np.sum((res_exp/sigma)**2)
    chi2_bi  = np.sum((res_bi/sigma)**2)

    chi2red_exp = chi2_exp/(N-k_exp)
    chi2red_bi  = chi2_bi/(N-k_bi)

    aic_exp = N*np.log(rss_exp/N) + 2*k_exp
    aic_bi  = N*np.log(rss_bi/N) + 2*k_bi

    delta_aic = aic_exp - aic_bi

    print(f'H0 = {H0[i]:.1f} kA/m')
    print(f'Exp:   tau = {tau:6.1f} s | RMSE = {rmse_exp:6.3f} °C | R² = {r2_exp:7.5f} | χ²ν = {chi2red_exp:8.2f} | AIC = {aic_exp:8.1f}')
    print(f'BiExp: tau1 = {tau1:6.1f} s | tau2 = {tau2:6.1f} s | RMSE = {rmse_bi:6.3f} °C | R² = {r2_bi:7.5f} | χ²ν = {chi2red_bi:8.2f} | AIC = {aic_bi:8.1f}')
    print(f'ΔAIC = {delta_aic:.1f}')
    print('-'*100)

    Taux.append(T_aux)
    fits_exp.append(Tfit_exp)
    fits_biexp.append(Tfit_bi)
    residuos_exp.append(res_exp)
    residuos_biexp.append(res_bi)

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(8,5),constrained_layout=True)

    ax1.plot(t_aux-t_aux[0],T_aux,'.',ms=3,label='Datos')
    ax1.plot(t_aux-t_aux[0],Tfit_exp,'-',lw=2,label='Exp.')
    ax1.plot(t_aux-t_aux[0],Tfit_bi,'--',lw=2,label='Bi-exp.')

    texto = (rf'$\chi^2_{{\nu,\mathrm{{exp}}}}={chi2red_exp:.2f}$' '\n'
             rf'$\chi^2_{{\nu,\mathrm{{bi}}}}={chi2red_bi:.2f}$' '\n'
             rf'$\Delta AIC={delta_aic:.1f}$')

    ax1.text(0.80,0.50,texto,
             transform=ax1.transAxes,
             ha='left',
             va='top',
             bbox=dict(boxstyle='round',
                       facecolor='white',
                       alpha=0.85))

    ax1.set_title('Temperatura vs tiempo',loc='left')
    ax1.set_ylabel('T (°C)')
    ax1.set_xlabel('t (s)')

    ax2.set_title('Residuos vs temperatura',loc='left')
    ax2.plot(T_aux,res_exp,'.-',label='Residuo exp.')
    ax2.plot(T_aux,res_bi,'.-',label='Residuo bi-exp.')
    ax2.axhline(0,color='k',lw=1)

    ax2.set_xlim(-175,0)
    ax2.set_xlabel('T (°C)')
    ax2.set_ylabel('Residuo (°C)')

    for a in [ax1,ax2]:
        a.grid()
        a.legend()

    plt.suptitle(f'4.5 - EG 51% FF 49% - LN2 → RF - H$_0$ = {H0[i]:.1f} kA/m')

#%% Ploteo todos los residuos
fig46,ax = plt.subplots(figsize=(8,4),constrained_layout=True)
ax.set_title('Residuos vs Temp - EG 51% FF 49%',loc='left')

for i in range(len(residuos_biexp)):
    ax.plot(Taux[i],residuos_biexp[i],'.-',label=f'{H0[i]:.1f}')

ax.axhline(0,color='k',lw=1)
ax.axvspan(-155,-130,color='tab:red',alpha=0.2,zorder=-1)
ax.axvspan(-70,-40,color='tab:orange',alpha=0.2,zorder=-1)

ax.grid()
ax.legend(title='H$_0$ (kA/m)',loc='lower right',ncol=1)
ax.set_xlim(-170,0)
ax.set_ylim(-10,10)
#%%
fig4.savefig('4_EG51FF49_LN2_RF_150_125_100_075_050_000.png',dpi=300)
fig43.savefig('4_EG51FF49_grad_temp_150_125_100.png',dpi=300)
fig44.savefig('4_EG51FF49_grad_temp_075_050_000.png',dpi=300)
fig45.savefig('4_EG51FF49_grad_temp_150_125_100_075_050_000.png',dpi=300)
fig46.savefig('4_EG51FF49_templogs_residuos.png',dpi=300)
#%% 19 Junio 
''' Ahora pruebo usando del filtro Savitzky-Golay: Ajusta un polinomio local y deriva el polinomio.
'''
from scipy.signal import savgol_filter
_,t_2_57, T_2_57,_ = lector_templog(temps_500_EG55_FF45_2[0])
_,t_2_20, T_2_20,_ = lector_templog(temps_500_EG55_FF45_2[-2])

dTdt_2_57 = savgol_filter(T_2_57,window_length=11,polyorder=3,deriv=1,delta=1.0)
dTdt_2_20 = savgol_filter(T_2_20,window_length=11,polyorder=3,deriv=1,delta=1.0)

dTTdt_2_57 = savgol_filter(T_2_57,window_length=11,polyorder=3,deriv=2,delta=1.0)
dTTdt_2_20 = savgol_filter(T_2_20,window_length=11,polyorder=3,deriv=2,delta=1.0)

curv_2_57 = np.abs(dTTdt_2_57)/(1+dTdt_2_57**2)**1.5
curv_2_20 = np.abs(dTTdt_2_20)/(1+dTdt_2_20**2)**1.5

fig,((a,b),(c,d),(e,f),(g,h))= plt.subplots(nrows=4,ncols=2,figsize=(15,10),sharex='col',sharey=False,constrained_layout=True)

a.plot(t_2_57,T_2_57,'.-',c='C0',label='T')
b.plot(t_2_20,T_2_20,'.-',c='C1',label='T')

c.plot(t_2_57,dTdt_2_57,'.-',c='C0',label='SG dT/dt')
c.plot(t_2_57,np.gradient(T_2_57,t_2_57),'.-',c='C2',alpha=0.7,label='np.grad dT/dt')

d.plot(t_2_20,dTdt_2_20,'.-',c='C1',label='SG dT/dt')
d.plot(t_2_20,np.gradient(T_2_20,t_2_20),'.-',c='C3',alpha=0.7,label='np.grad dT/dt')

e.plot(t_2_57,dTTdt_2_57,'.-',c='C0',label='SG d²T/dt')
e.plot(t_2_57,np.gradient(np.gradient(T_2_57,t_2_57),t_2_57),'.-',c='C2',alpha=0.7,label='np.grad d²T/dt²')

f.plot(t_2_20,dTTdt_2_20,'.-',c='C1',label='SG d²T/dt²')
f.plot(t_2_20,np.gradient(np.gradient(T_2_20,t_2_20),t_2_20),'.-',c='C3',alpha=0.7,label='np.grad d²T/dt²')

g.plot(t_2_57,curv_2_57,'.-',c='C0',label='Curvatura')
h.plot(t_2_20,curv_2_20,'.-',c='C1',label='Curvatura')

a.set_title('57 kA/m')
b.set_title('20 kA/m')
b.set_ylim(-175,-100)
f.set_ylim(-0.4,0.4)
for m in [a,c,e,g]:
    m.set_xlim(80,130)
    m.grid()
    m.legend()

for n in [b,d,f,h]:
    n.set_xlim(80,200)
    n.grid()
    n.legend()

for n in [g,h]:
    n.set_xlabel('t (s)')
    
plt.suptitle('EG 55% FF 45% - LN2 --> RF - Idc = [150, 050] dA')    
plt.savefig('nueva_comparativa_templogs.png',dpi=300)
# %%


#%% 2- EG 55% FF 45% - LN2 --> RF - Idc = [150 to 000] dA

# %% 3- 
