dx = 2.22e-6; % The pixel size in m

dps = [22.5e-6:.5e-6:40e-6]; % The particle sizes to test with jiangPSize (specify dpmin, ddp and dpmax - I would not go any smaller than 1/4 pixel)

% Use the same bins and the same dz as in the classic reconstruction!

% Use particlemetrics from guihandles.ps for all subsequent steps!
% load('mymetricsfile.mat');
% Mymetrisfile must contain the following variables from GUIhandles.ps and
% the hist.mat files:
% 
ps = GUIhandles.ps; 

metricnames = ps.prtclmetricvarnames;

files = dir('matfiles_3.0/whole/*hist.mat');
files = {files(:).name};
% 
% Finally, select only the particles you want to have (e.g. everything
% classified as Particle_round by hand)
% Example: 
% idx2a = find(ps.isparticlebyhand == 'Particle_round');
% idx2b = find(ps.isparticlebyhand == 'Particle_nubbly');
% idx = union(idx2a,idx2b);
idx = find(ps.isparticlebypredict=='Particle_round');

holonum = ps.holonum(idx);
prtclIm = ps.prtclIm(idx);

metrics = ps.prtclmetrics(idx,:);

xpos = metrics(:,(strcmp(metricnames,{'xpos'})));
ypos = metrics(:,(strcmp(metricnames,{'ypos'})));
zpos = metrics(:,(strcmp(metricnames,{'zpos'})));
majsiz = metrics(:,(strcmp(metricnames,{'majsiz'})));
minsiz = metrics(:,(strcmp(metricnames,{'minsiz'})));

existing_parts = dir('jps_part');
if ~isempty(existing_parts)&&~(existing_parts(end).isdir)
    start_pid = str2double(existing_parts(end).name(10:12));
else
    start_pid = 1;
end

%% Sweep over all particles

for pID = start_pid:length(xpos)

fn = files{holonum(pID)}; % Get the file name of the actual hologram
fn = config.replaceEnding(fn,'_hist.mat','.png');

% Get position and size for the actual particle
tempx = xpos(pID);
tempy = ypos(pID);
tempz = zpos(pID);
tempsl = minsiz(pID);
tempsu = majsiz(pID);

% Estimated z position +/- 1 mm is the range where we search for the
% focal plane. The spacing is 100 um. Choose whatever fits your needs. 
zs = tempz-1e-3:1e-4:tempz+1e-3;

% Get the config object and call jiangPSize
cfg = config('holoviewer.cfg');
jps = jiangPSize(cfg);

% Get the particle number as string
no = num2str(1e3+pID);
no = no(2:end);
sfn2 = ['jps_part/particle_',no,'.mat']; % Filename to save

% Do the filtered reconstruction for the selected particle
jps.doMoreEfficientPartSweep(fn,tempx,tempy,tempz,sqrt(tempsl*tempsu), zs, dps, sfn2);

disp(['Done with ', num2str(pID), ' out of ', num2str(length(xpos)), ' particles.'])

end
