import pandas as pd


OBJECT_TYPE_MAP = {
    0: "pedestrian",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "group_of_ped",
    5: "bus",
    6: "road_crossing",
    7: "truck",
    8: "road_work",
    9: "traffic_light",
    10: "rider",
    11: "stop_sign",
    16: "dog",
    17: "horse",
}


""" 
Visible Object Mapping
"""
VISIBLE_OBJECT_SIDE_MAP = {
    0: 'UNK',
    1: 'front_side',
    2: 'front_left_side',
    3: 'front_right_side',
    4: 'rear_side',
    5: 'rear_right_side',
    6: 'rear_left_side',
    7: 'left_side',
    8: 'right_side',
}

""" 
Tailights Status Mapping
"""
REAR_LIGHT_STATUS_10_MAP = {
    0: 'UNK',
    1: 'BOO',
    2: 'OLO',
    3: 'OLR',
    4: 'OOO',
    5: 'OOR',
    6: 'BLO',
    7: 'BLR',
    8: 'BOR',
    9: 'REVERSE',
}

REAR_LIGHT_STATUS_6_MAP = {
    0: 'UNK',
    1: 'BOO',
    2: 'OLO',
    3: 'OLR',
    4: 'OOO',
    5: 'OOR',
    6: 'OLO',
    7: 'OLR',
    8: 'OOR',
    9: 'REVERSE',
}



def get_hazard_classes_stat(df):

    video_df = get_video_level_df(df)

    total_videos = video_df["video_n"].nunique()
    
    video_df = video_df.rename(columns={
        #"num_videos": "N. of Samples",
        "hazard_type_name": "Class Name"
    })

    stat = (
        video_df.groupby("Class Name")
        .size()
        .reset_index(name="N. of Samples")
        .sort_values("N. of Samples", ascending=False)
    )

    return stat


def get_video_level_df(df):

    video_df = (
        df[["video_n", "hazard_type_name"]]
        .drop_duplicates(subset=["video_n"])   # keep only one row per video
        .reset_index(drop=True)
    )

    return video_df


def get_object_classes_stat(df):

    # keep only target objects
    df = df[df["ID"] == 0].copy()

    # map int → string
    df["Class Name"] = df["object_type"].map(OBJECT_TYPE_MAP)

    stat = (
        df.groupby("Class Name")
        .size()
        .reset_index(name="N. of Samples")
        .sort_values("N. of Samples", ascending=False)
    )

    return stat


def get_object_visible_side_classes_stat(df):

    # keep only target objects
    df = df[df["ID"] == 0].copy()

    df = df.rename(columns={
        "object_visible_side": "Class Name"
    })
    
    stat = (
        df.groupby("Class Name")
        .size()
        .reset_index(name="N. of Samples")
        .sort_values("N. of Samples", ascending=False)
    )

    return stat


def get_rear_light_status_classes_stat(df):

    # keep only target objects
    df = df[df["ID"] == 0].copy()

    df = df.rename(columns={
        "tailight_status": "Class Name"
    })
    
    stat = (
        df.groupby("Class Name")
        .size()
        .reset_index(name="N. of Samples")
        .sort_values("N. of Samples", ascending=False)
    )

    return stat



def get_rear_light_status_classes_stat(df):

    # keep only target objects
    df = df[df["ID"] == 0].copy()

    # map labels
    df["RVSR-10"] = df["tailight_status_int"].map(REAR_LIGHT_STATUS_10_MAP)
    df["RVSR-6"] = df["tailight_status_int"].map(REAR_LIGHT_STATUS_6_MAP)

    # counts for each taxonomy
    stat10 = (
        df.groupby("RVSR-10")
        .size()
        .reset_index(name="RVSR-10 Samples")
    )

    stat6 = (
        df.groupby("RVSR-6")
        .size()
        .reset_index(name="RVSR-6 Samples")
    )

    # merge into single table
    stat = (
        stat10.merge(
            stat6,
            left_on="RVSR-10",
            right_on="RVSR-6",
            how="outer"
        )
    )

    stat["RVSR-6 Samples"] = (
        stat["RVSR-6 Samples"]
        .fillna(0)
        .astype(int)
    )

    # optional: nicer column ordering
    stat = stat[["RVSR-10", "RVSR-10 Samples", "RVSR-6 Samples"]]

    return stat