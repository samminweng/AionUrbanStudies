// Draw the term chart and document list
function TermChart(cluster, cluster_docs){
    let keyword_clusters = cluster['KeywordClusters'];
    let total_keywords = keyword_clusters.reduce((pre, cur) =>{
        return pre + cur['Key-phrases'].length;
    }, 0);



    // Create a cluster heading
    function createClusterHeading(){
        const cluster_no = cluster['Cluster'];
        const heading = $('<div class="mt-3"><span class="h5">Article Cluster ' + cluster_no +'</span></div>');
        const cluster_link = $('<span type="button" class="btn btn-link" >' + cluster_docs.length + ' articles</span>');
        // Define onclick event cluster link
        cluster_link.click(function(){
            const doc_list = new DocList(cluster_docs, [], "");
            document.getElementById('doc_list').scrollIntoView({behavior: "smooth",
                block: "nearest", inline: "nearest"});
        });
        heading.append(cluster_link);
        heading.append($('<span>' + total_keywords + ' keywords</span>'));

        $('#cluster_heading').empty();
        $('#cluster_heading').append(heading);
    }


    // Main entry
    function createUI(){
        try{
            $('#term_occ_chart').empty();
            $('#sub_group').empty();
            $('#doc_list').empty();
            $('#key_phrase_chart').empty();
            // Display bar chart to show groups of key phrases as default graph
            const graph = new BarChart(keyword_clusters, cluster, cluster_docs);
            createClusterHeading();
        }catch (error){
            console.error(error);
        }
    }
    createUI();
}
