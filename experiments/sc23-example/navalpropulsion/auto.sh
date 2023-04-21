#!/bin/bash

export NUM_WORKERS=8
export timeout=70
export DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME=1 
export DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME=0.01
export random_states=(1608637542 3421126067 4083286876  787846414 3143890026 3348747335 2571218620 2563451924  670094950 1914837113) 

exec_experiments () {
    ./dhb_navalpropulsion-CBO-DUMMY-UCB.sh
    ./dhb_navalpropulsion-CBO-RF-UCB.sh
    ./dhb_navalpropulsion-CBO-RF-UCB-SHA.sh
    ./dhb_navalpropulsion-DBO-RF-UCB.sh
    ./dhb_navalpropulsion-DBO-RF-UCB-SHA.sh
}

printf "Creating 'output' and 'figures' directories if not exist.\n"
mkdir -p output
mkdir -p figures

printf "Reconfiguring 'plot.yaml' file.\n"
sed -i '' "s_data-root: .*_data-root: $PWD/output_" plot.yaml
sed -i '' "s_figures-root: .*_figures-root: $PWD/figures" plot.yaml

printf "Executing experiments.\n"
for random_state in ${random_states[@]}; do
    export random_state=$random_state;
    printf "Executing experiment serie with random_state: $random_state\n"
    for i in {1..5}; do 
        exec_experiments && break || sleep 5;
    done
    sleep 1;
done

printf "Plotting results.\n"
python -m scalbo.plot --config plot.yaml