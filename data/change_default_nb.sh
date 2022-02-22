#!/bin/bash

nb_states="$1"
states_dir="all_states/states_$nb_states"

if [ ! -d "$states_dir" ]
then
    echo "These states have not been generated before"
    exit 1
fi

# Modify `states/` symbolic link
ln -sfn "$states_dir" states

for dir in runs/**/AMN_*/*/* runs/**/Transfer_*/day_0/ runs/**/Transfer_*/day_1/*
do

    if [ ! -f "${dir}/actions_${nb_states}.npy" ]
    then
        echo "No actions found in ${dir}"
        rm "${dir}/actions.tsv" "${dir}/features.tsv" 2&>/dev/null
        continue
    fi

    if [ ! -f "${dir}/features_${nb_states}.npy" ]
    then
        echo "No features found in ${dir}"
        rm "${dir}/actions.tsv" "${dir}/features.tsv" 2&>/dev/null
        continue
    fi

    if [ ! -f "${dir}/actions_${nb_states}.tsv" -o ! -f "${dir}/features_${nb_states}.tsv" ]
    then
        echo "The actions or features have not been converted to tab format. Converting..."
        ./to_tab.py "$dir"
        echo "Done!"
    fi

    ln -sf "actions_${nb_states}.npy" "${dir}/actions.npy"
    ln -sf "features_${nb_states}.npy" "${dir}/features.npy"

    ln -sf "actions_${nb_states}.tsv" "${dir}/actions.tsv"
    ln -sf "features_${nb_states}.tsv" "${dir}/features.tsv"

    ln -sf "qvalues_${nb_states}.npy" "${dir}/qvalues.npy"
done
