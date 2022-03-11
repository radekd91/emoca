from gdl.datasets.AffectNetDataModule import AffectNetDataModule



def main(): 
    if len(sys.argv) < 2: 
        print("Usage: python process_affectnet.py <input_folder> <output_folder> <optional_processed_subfolder> <optional_subset_index>")
        print("input_folder ... folder where you downloaded and extracted AffectNet")
        print("output_folder ... folder where you want to process AffectNet")
        print("optional_processed_subfolder ... if AffectNet is partly processed, it created a subfolder, which you can specify here to finish processing")
        print("optional_subset_index ... index of subset of AffectNet if you want to process many parts in parallel (recommended)")

    downloaded_affectnet_folder = sys.argv[1]
    processed_output_folder = sys.argv[2]

    if len(sys.argv) >= 3: 
        processed_subfolder = sys.argv[3]
    else: 
        processed_subfolder = None


    if len(sys.argv) >= 4: 
        sid = int(sys.argv[4])
    else: 
        sid = None


    dm = AffectNetDataModule(
            downloaded_affectnet_folder,
             "/is/cluster/work/rdanecek/data/affectnet/",
             processed_subfolder=processed_subfolder,
             mode="manual",
             scale=1.25,
             ignore_invalid=True,
            )

    if sid is not None:
        if sid >= dm.num_subsets: 
            print("Subset index is larger than number of subsets")
            sys.exit()
        dm._detect_landmarks_and_segment_subset(dm.subset_size * sid, min((sid + 1) * dm.subset_size, len(dm.df)))
    else:
        dm.prepare_data() 

    
    


if __name__ == "__main__":
    main()

