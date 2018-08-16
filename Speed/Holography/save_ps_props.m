function [xp, yp, zp, times] = save_ps_props(ps, folder)

try
    rrp = load([folder, '/removeRepeatedPart.mat']);
    ps_rem = rrp.remove;
    sprintf('Loaded from file.')
catch
    ps_rem = ps.removeMyRepeatedParticles(ps, 5e-6);
end

% selector = and(ps_rem.sI, ps.isparticlebyhand=='Particle_nubbly');
% selector = and(ps_rem.sI, (ps.isparticlebypredict=='Particle_nubbly')|(ps.isparticlebypredict=='Aggregate')|(ps.isparticlebypredict=='Dendrite'));
% selector = and(ps_rem.sI, (ps.isparticlebypredict=='Particle_nubbly')...
%     |(ps.isparticlebypredict=='Plate')...
%     |(ps.isparticlebypredict=='Aggregate')|(ps.isparticlebypredict=='Needle'));
selector = and(ps_rem.sI, ps.isparticlebypredict=='Particle_round');

xp = ps.xpos(selector);
yp = ps.ypos(selector);
zp = ps.zpos(selector);

prediction = char(ps.isparticlebypredict(selector));

times = ps.holonum(selector);
area = ps.area(selector);

majsiz = ps.majsiz(selector);
minsiz = ps.minsiz(selector);

ims = ps.prtclIm(selector);

save([folder,'/ps_bypredict.mat'], 'xp', 'yp', 'zp', 'prediction', 'times', 'area', 'majsiz', 'minsiz', 'ims');

