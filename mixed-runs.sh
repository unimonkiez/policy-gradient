numberOfEpisodes=10000
bOptions=(1 5)
RTGOptions=(True False)
baselineOptions=(True False)

for b in ${bOptions[*]}
do
    N=$((numberOfEpisodes / b))
    for RTG in ${RTGOptions[*]}
    do
        for baseline in ${baselineOptions[*]}
        do
            msg="**************************************************\n"
            msg+="b=${b} (N=${N}), RTG=${RTG}, baseline=${baseline}\n"
            msg+="**************************************************\n"            
            echo "$msg"
            
            args="--b ${b} --N ${N}"
            if [ $RTG = "True" ]; then
                args+=" --RTG"
            fi
            if [ $baseline = "True" ]; then
                args+=" --baseline"
            fi

            ./venv/bin/python -m src ${args} >run_b${b}_RTG${RTG}_baseline${baseline}.out 2>&1 &
        done
    done
done
wait
