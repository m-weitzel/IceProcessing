% load('mymetricsfile.mat');

if ~exist('majsiz_new', 'var')
    
    ps = GUIhandles.ps;
    idx = find(ps.isparticlebypredict=='Particle_round');

    holonum = ps.holonum(idx);
    prtclIm = ps.prtclIm(idx);

    metricnames = ps.prtclmetricvarnames;
    metrics = ps.prtclmetrics(idx,:);

    files = dir('matfiles_3.0/whole/*hist.mat');
    files = {files(:).name};
    old_majsiz = metrics(:,(strcmp(metricnames,{'majsiz'})));
    
    flist_jps = dir('jps_part');
    N = length(flist_jps)-2;
    majsiz_new = zeros(N, 1);
    for i=1:N
        load(['jps_part/particle_', num2str(i, '%03d'), '.mat']);
        maxAtDs = max(maxamp,[],2);
        [~,ml2] = max(maxAtDs);
        majsiz_new(i)=dps(ml2);
    end
end
    
bins=linspace(20e-6, 45e-6, 30);

save('histogram_data', 'old_majsiz', 'majsiz_new', 'bins');
    
h_alt = hist(old_majsiz, bins);
h_neu = hist(majsiz_new, bins);

figure()
b_p = bar([h_alt', h_neu'], 'grouped', 'BarWidth', 1.5);
lgd = legend(b_p, 'Initial', 'JPS');
lgd.FontSize=16;

xd = findobj('-property','XData');

for i=1:2
    dat = get(xd(i),'XData');
    set(xd(i),'XData',bins);
end

xlabel('Diameter in \mum', 'FontSize',16)
ylabel('Count', 'FontSize', 16)
xt = get(gca, 'XTick');
set(gca, 'FontSize', 16)

xlim([20e-6 45e-6])
