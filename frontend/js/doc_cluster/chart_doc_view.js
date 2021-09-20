function ChartDocView(total_clusters, doc_cluster_data) {
    function _createUI() {
         // Create the nodes and links of term occurrences.
        let d3_scatter_graph = new ScatterGraph(total_clusters, doc_cluster_data);
        // Create a summary list view for
        let cluster_list_view = new ClusterListView(total_clusters);
    }

    _createUI();
}
