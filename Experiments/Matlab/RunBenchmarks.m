% orkut - 1.78GB
X = table2array(readtable("orkut-svs.txt", "Format", "%u%u"));
Y = table2array(readtable("orkut-Y50-sparse.txt", "Format", "%d"));
f = @() GraphEncoder(X,Y);
timeit(f)


% Twitch - 80MB
X = table2array(readtable("twitch_edges.csv", "Format", "%u%u"));
Y = table2array(readtable("twitchFullY-20-removed.txt", "Format", "%d"));
f = @() GraphEncoder(X,Y);
timeit(f)


% LiveJournal - 1GB
X = table2array(readtable("soc-LiveJournal1.txt", "Format", "%u%u"));
Y = table2array(readtable("liveJournal-Y50-sparse.txt", "Format", "%d"));
f = @() GraphEncoder(X,Y);
timeit(f)


% pokec - 478MB
X = table2array(readtable("soc-pokec-SNAP.txt", "Format", "%u%u"));
Y = table2array(readtable("pokec-Y50-sparse.txt", "Format", "%d"));
f = @() GraphEncoder(X,Y);
timeit(f)


% % orkut-groups - 5.1GB
% X = table2array(readtable("aff-orkut-user2groups.edges", "Format", "%u%u"));
% Y = table2array(readtable("orkut-groups-Y40-sparse.txt", "Format", "%d"));
% f = @() GraphEncoder(X,Y);
% timeit(f)