n_procs=94 
echo "Will spawn ${n_procs} processes to download EmotionNet dataset"
for i in ` seq 0 ${n_procs}`; do
	echo "Spawning process ${i} "
    python emotion_net_downloader.py ${i} &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done

echo "All processes finished."