import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt


x_qual = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
x_tip  = np.arange(0, 11, 1)


temp_good = fuzz.trimf(x_qual, [-1, 0, 1])
temp_control = fuzz.trimf(x_qual, [0, 1, 3])
temp_crit = fuzz.trapmf(x_qual, [1, 2, 3,4])
temp_crush = fuzz.trapmf(x_qual, [3, 4, 9,10])
oil_good = fuzz.trimf(x_serv, [-1, 0, 2])
oil_control = fuzz.trapmf(x_serv, [1,2,6,7])
oil_crush = fuzz.trimf(x_serv, [6,10,11])
con_perf = fuzz.trimf(x_tip, [0, 2, 4])
con_tocheck = fuzz.trimf(x_tip, [2, 4, 6])
con_tocheckmonth = fuzz.trimf(x_tip, [4, 6, 8])
con_switchoff = fuzz.trimf(x_tip, [6, 8, 10])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_qual, temp_good, 'b', linewidth=1.5, label='good')
ax0.plot(x_qual, temp_control, 'g', linewidth=1.5, label='control')
ax0.plot(x_qual, temp_crit, 'r', linewidth=1.5, label='crit')
ax0.plot(x_qual, temp_crush, 'm', linewidth=1.5, label='crush')
ax0.set_title('Temperature')
ax0.legend()

ax1.plot(x_serv, oil_good, 'b', linewidth=1.5, label='good')
ax1.plot(x_serv, oil_control, 'g', linewidth=1.5, label='control')
ax1.plot(x_serv, oil_crush, 'r', linewidth=1.5, label='crush')
ax1.set_title('Oil')
ax1.legend()

ax2.plot(x_tip, con_perf, 'r', linewidth=1.5, label='perfect')
ax2.plot(x_tip, con_tocheck, 'b', linewidth=1.5, label='tocheck')
ax2.plot(x_tip, con_tocheckmonth, 'g', linewidth=1.5, label='tocheckmonth')
ax2.plot(x_tip, con_switchoff, 'r', linewidth=1.5, label='switchoff')

ax2.set_title('Condition')
ax2.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()
plt.show()