class TermChartUtility {
    // Get the maximal doc length
    static get_max_node_size(nodes) {
        let max_node_size = 0;
        for (let node of nodes) {
            const doc_ids = node['doc_ids'];
            max_node_size = Math.max(max_node_size, doc_ids.length);
        }
        return max_node_size;
    }

    // Convert the collocations json to the format of D3 network graph
    static create_node_link_data(searched_term, term_map) {
        // Populate the nodes with collocation data
        let nodes = [];
        // Add other nodes from term map
        for (let i = 0; i < term_map.length; i++) {
            const tm = term_map[i];
            const col_name = tm[0];
            const doc_ids = tm[1];
            const group = (col_name === searched_term) ? 0: 1;
            const node = {'id': i, 'name': col_name, 'doc_ids': doc_ids, 'group': group};
            nodes.push(node);
        }
        // Populate the links with occurrences
        let links = [];
        let source = 0;
        for (let target = 1; target < nodes.length; target++) {
            const occ = nodes[target]['doc_ids'].length;
            // Add the link
            links.push({source: source, target: target, value: occ});
        }
        return {nodes: nodes, links: links};
    }

    // Filter term map by the year
    static filter_term_map(documents, term_map) {
        // let filter_doc_ids = documents.filter(doc => doc['Year'] <= year).map(doc => doc['DocId']);
        const doc_ids = documents.map(doc => doc['DocId']);
        console.log(doc_ids);
        let filtered_term_map = [];
        // Filter out term map
        for (const tm of term_map) {
            const collocation = tm[0];
            const filtered_doc_ids = tm[1].filter(doc_id => doc_ids.includes(doc_id));
            if (filtered_doc_ids.length > 0) {
                filtered_term_map.push([collocation, filtered_doc_ids]);
            }
        }
        console.log(filtered_term_map);
        return filtered_term_map
    }

    // Filter the documents by key terms
    static filter_documents_by_key_terms(searched_term, complementary_terms, term_map, documents) {
        // console.log(term_map);
        let doc_id_s = term_map.find(term => term[0] === searched_term)[1];
        let intersection = new Set(doc_id_s);
        for (let complementary_term of complementary_terms) {
            let doc_id_c = term_map.find(term => term[0] === complementary_term)[1];
            intersection = new Set([...intersection].filter(doc_id => doc_id_c.includes(doc_id)));
        }
        console.log(intersection);
        const filtered_documents = documents.filter(doc => intersection.has(doc['DocId']));
        // Sort the document by year
        filtered_documents.sort((a, b) => b['Year'] - a['Year']);
        return filtered_documents;
    }


}



