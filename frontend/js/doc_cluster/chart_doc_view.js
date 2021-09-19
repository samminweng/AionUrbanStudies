function ChartDocView(total_clusters, doc_cluster_data) {
    function _createUI() {
         // Create the nodes and links of term occurrences.
        let d3_scatter_graph = new ScatterGraph(total_clusters, doc_cluster_data);
        // // Create a document list view for a collocation
        // let doc_list_view = new DocumentListView(key_terms, collocation_data, corpus_data);
    }

    _createUI();
}
