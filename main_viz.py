from visualization.data_loading import load_dataset, keep_target_objects
from visualization.feature_viz import (
    plot_hazard_distribution,
    plot_unknown_values,
    plot_tailight_vs_hazard,
    plot_visible_side_vs_hazard,
    plot_corr_among_numeric_features,
    plot_corr_between_numeric_features_and_continuous_targets,
    plot_corr_between_categorical_features_and_categorical_targets,
    plot_corr_between_numeric_features_and_binary_target,
    
    cramers_v_plot,
    plot_feature_vs_hazard,
)
from visualization.collect_dataset_statistics import (
    get_hazard_classes_stat,
    get_object_classes_stat,
    get_object_visible_side_classes_stat,
    get_rear_light_status_classes_stat,
    get_frame_statistics
)
from visualization.image_viz import (
    show_image_with_bbox
)
from visualization.model_output_viz import (
    plot_predictions_vs_gt
)
from visualization.preprocessing_viz import (
    plot_bbox_size_distribution,
    plot_object_positions
)
from visualization.temporal_viz import (
    plot_hazard_over_time,
    plot_speed
)
from visualization.target_analysis import (
    plot_hazard_balance
)

from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages



spatial_features = [
    "xc", "yc", "w", "h", "bbox_area",
    "x_1", "y_1", "x_2", "y_2",
    "x_n", "y_n", "w_n", "h_n", "bbox_area_n"
]

motion_features = [
    "vx_n", "vy_n",
    "ax_n", "ay_n",
    "speed",
    "theta", "dtheta"
]

shape_features = [
    "scale", "dscale",
    "aspect", "daspect",
    "border_dist"
]

categorical_numeric = [
    "object_type",
    "rear_light_status_int",
    "object_visible_side_int",
]

all_numeric_features = (
    spatial_features
    + motion_features
    + shape_features
)

dataset_csv_path = Path(r"C:\Projects\RoadHazardDataset\frame_sequences\all_roadHazardDataset_videos.csv")

dataset_df = load_dataset(dataset_csv_path)

dataset_df = keep_target_objects(dataset_df)

#----------------------------#
# Feture Visualizations
#----------------------------#

with PdfPages("./output/visualizations/eda_correlation_report.pdf") as pdf:
    
    stat = get_hazard_classes_stat(dataset_df)
    
    latex = stat.to_latex(
        index=False,
        float_format="%.1f",
        caption="Distribution of hazard classes at video level.",
        label="tab:number_of_samples_per_classes",
        column_format="lcc"
    )
    
    
    latex = latex.replace(
        r"\centering",
        r"\centering" + "\n\\begin{small}"
    )
    
    latex = latex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n\\end{small}"
    )
    
    with open("C:/Manuscripts/Hazard Recognition Paper/tab_hazard_classes_samples.tex", "w") as f:
        f.write(latex)
    
    stat = get_object_classes_stat(dataset_df)
    
    latex = stat.to_latex(
        index=False,
        float_format="%.1f",
        caption="Distribution of object classes.",
        label="tab:number_of_samples_per_object_classes",
        column_format="lcc"
    )
    
    latex = latex.replace(
        r"\centering",
        r"\centering" + "\n\\begin{small}"
    )
    
    latex = latex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n\\end{small}"
    )
    
    with open("C:/Manuscripts/Hazard Recognition Paper/tab_target_objects_classes_samples.tex", "w") as f:
        f.write(latex)
    
    stat = get_object_visible_side_classes_stat(dataset_df)
    
    latex = stat.to_latex(
        index=False,
        float_format="%.1f",
        caption="Number of samples per available classes for the target's object visible side datasets.",
        label="tab:THD:ObjectVisibleSide_Samples_per_class",
        column_format="lcc"
    )
    
    latex = latex.replace(
        r"\centering",
        r"\centering" + "\n\\begin{small}"
    )
    
    latex = latex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n\\end{small}"
    )
    
    with open("C:/Manuscripts/Hazard Recognition Paper/tab_object_visible_side_samples.tex", "w") as f:
        f.write(latex)
        
    stat = get_rear_light_status_classes_stat(dataset_df)
    
    latex = stat.to_latex(
        index=False,
        #float_format="%.1f",
        caption="Number of samples per available classes for the RVSR-6 and RVSR-10 datasets.",
        label="tab:THD:Samples_per_class",
        column_format="lccc"
    )
    
    latex = latex.replace(
        r"\centering",
        r"\centering" + "\n\\begin{small}"
    )
    
    latex = latex.replace(
        r"\end{tabular}",
        r"\end{tabular}" + "\n\\end{small}"
    )
    
    with open("C:/Manuscripts/Hazard Recognition Paper/tab_rear_light_status_samples.tex", "w") as f:
        f.write(latex)
    
    
    avg_class, overall = get_frame_statistics(dataset_df)

    print(avg_class)
    print(overall)
    
    fig1, ax = plot_hazard_distribution(dataset_df)
    pdf.savefig(fig1)
    
    fig2, ax = plot_unknown_values(dataset_df)
    pdf.savefig(fig2)
    
    fig3, ax = plot_tailight_vs_hazard(dataset_df)
    pdf.savefig(fig3)
    
    fig4, ax = plot_visible_side_vs_hazard(dataset_df)
    pdf.savefig(fig4)
    
    result_df, fig5, ax = cramers_v_plot(
        dataset_df,
        categorical_numeric,
        target = "hazard_flag"
    )
    pdf.savefig(fig5)
    print("cramers_v_plot:", result_df)
    
    result_df, fig14, ax = cramers_v_plot(
        dataset_df,
        categorical_numeric,
        target = "hazard_flag"
    )
    pdf.savefig(fig14)
    print("cramers_v_plot:", result_df)
    
    feature = "speed"
    fig6, ax = plot_feature_vs_hazard(
        dataset_df,
        feature,
        target="hazard_type_int"
    )
    pdf.savefig(fig6)
    
    # Correlation among features using spearman matrix (pearson, spearman, kendall)
    fig7, ax = plot_corr_among_numeric_features(
        dataset_df,
        all_numeric_features,
        method="spearman",
        title = "all_numeric_features"
    )
    pdf.savefig(fig7)
    
    # Correlation between features and target. Point-Biserial Correlation method which is used when input vars are continuous and target var is binary
    result_df, fig8, ax = plot_corr_between_numeric_features_and_binary_target(
        dataset_df,
        all_numeric_features,
        title = "all_numeric_features"
    )
    pdf.savefig(fig8)
    print("plot_corr_between_numeric_features_and_binary_target:", result_df)
    
    # Correlation between features and target. Pearson Correlation Feature which uses continuos input features and continuous target (not ideal for the project)
    fig9, ax = plot_corr_between_numeric_features_and_continuous_targets(
        dataset_df,
        all_numeric_features,
        target = "hazard_flag",
        method="pearson",
        title="all_numeric_features"
    )
    pdf.savefig(fig9)
    
    chi2_df, fig10, ax = plot_corr_between_categorical_features_and_categorical_targets(
        dataset_df,
        categorical_numeric,
        target = "hazard_flag",
        method="chi2",
        title="hazard_flag"
    )
    print("chi2_df:", chi2_df)
    pdf.savefig(fig10)
    
    mi_df, fig11, ax = plot_corr_between_categorical_features_and_categorical_targets(
        dataset_df,
        categorical_numeric,
        target="hazard_flag",
        method="mutual_info",
        title="hazard_flag"
    )
    print("mi_df:", mi_df)
    pdf.savefig(fig11)
    
    chi2_df, fig12, ax = plot_corr_between_categorical_features_and_categorical_targets(
        dataset_df,
        categorical_numeric,
        target="hazard_type_int",
        method="chi2",
        title="hazard_type_int"
    )
    print("chi2_df:", chi2_df)
    pdf.savefig(fig12)
    
    mi_df, fig13, ax = plot_corr_between_categorical_features_and_categorical_targets(
        dataset_df,
        categorical_numeric,
        target="hazard_type_int",
        method="mutual_info",
        title="hazard_type_int"
    )
    print("mi_df:", mi_df)
    pdf.savefig(fig13)


#
#plot_bbox_size_distribution(dataset_df)
#
#plot_object_positions(dataset_df)
#
#plot_hazard_balance(dataset_df)
#
#plot_speed(dataset_df)
