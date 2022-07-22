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


qual_level_lo = fuzz.interp_membership(x_qual, temp_good, 2)
qual_level_md = fuzz.interp_membership(x_qual, temp_control, 2)
qual_level_hi = fuzz.interp_membership(x_qual, temp_crit,2)
qual_level_hi = fuzz.interp_membership(x_qual, temp_crush, 2)

serv_level_lo = fuzz.interp_membership(x_serv, oil_good,6.5)
serv_level_md = fuzz.interp_membership(x_serv, oil_control,6.5)
serv_level_hi = fuzz.interp_membership(x_serv, oil_crush, 6.5)


active_rule1 = np.fmax(qual_level_lo, serv_level_lo)


tip_activation_lo = np.fmin(active_rule1, con_perf)

tip_activation_md = np.fmin(serv_level_md, con_tocheck)

active_rule3 = np.fmax(qual_level_hi, serv_level_hi)
tip_activation_hi = np.fmin(active_rule3, con_switchoff)
tip0 = np.zeros_like(x_tip)


fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.fill_between(x_tip, tip0, tip_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(x_tip, con_perf, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(x_tip, tip0, tip_activation_md, facecolor='g', alpha=0.7)
ax0.plot(x_tip, con_tocheck, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, tip_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(x_tip, con_switchoff, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')


for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

aggregated = np.fmax(tip_activation_lo,
                     np.fmax(tip_activation_md, tip_activation_hi))



tip = fuzz.defuzz(x_tip, aggregated, 'mom')


tip_activation = fuzz.interp_membership(x_tip, aggregated, tip)  # for plot


fig, ax0 = plt.subplots(figsize=(8, 3))

ax0.plot(x_tip, con_perf, 'b', linewidth=0.5, linestyle='--', )
ax0.plot(x_tip, con_tocheck, 'g', linewidth=0.5, linestyle='--')
ax0.plot(x_tip, con_switchoff, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(x_tip, tip0, aggregated, facecolor='Orange', alpha=0.7)
ax0.plot([tip, tip], [0, tip_activation], 'k', linewidth=1.5, alpha=0.9)
ax0.set_title('Aggregated membership and result (line)')


for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()

x= x_tip

mfx = aggregated

defuzz_centroid = fuzz.defuzz(x, mfx, 'centroid')  # Same as skfuzzy.centroid
defuzz_bisector = fuzz.defuzz(x, mfx, 'bisector')
defuzz_mom = fuzz.defuzz(x, mfx, 'mom')
defuzz_som = fuzz.defuzz(x, mfx, 'som')
defuzz_lom = fuzz.defuzz(x, mfx, 'lom')

# Collect info for vertical lines
labels = ['centroid', 'bisector', 'mean of maximum', 'min of maximum',
          'max of maximum']
xvals = [defuzz_centroid,
         defuzz_bisector,
         defuzz_mom,
         defuzz_som,
         defuzz_lom]
colors = ['r', 'b', 'g', 'c', 'm']
ymax = [fuzz.interp_membership(x, mfx, i) for i in xvals]


plt.figure(figsize=(8, 5))

plt.plot(x, mfx, 'k')
for xv, y, label, color in zip(xvals, ymax, labels, colors):
    plt.vlines(xv, 0, y, label=label, color=color)
plt.ylabel('Fuzzy membership')
plt.xlabel('Universe variable (arb)')
plt.ylim(-0.1, 1.1)
plt.legend(loc=2)

plt.show()