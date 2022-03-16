// Utility functions
class Utility{
    // Create a dict to store the relation between word and doc
    static create_word_doc_dict(title_words, docs, word_key_phrase_dict){
        let dict = {};
        for(const [title_word, key_phrases] of Object.entries(word_key_phrase_dict)){
            let word_doc_ids = [];
            // Match the key phrases with title words and assign key phrase
            for(const key_phrase of key_phrases) {
                // // Collect the doc containing the key phrase
                const relevant_docs = docs.filter(d => {
                    const found = d['KeyPhrases'].find(kp => kp.toLowerCase() === key_phrase.toLowerCase());
                    if(found){
                        return true;
                    }
                    return false;
                });
                for (const doc of relevant_docs){
                    if(!word_doc_ids.includes(doc['DocId'])){
                        word_doc_ids.push(doc['DocId']);
                    }
                }
            }
            dict[title_word] = word_doc_ids;
        }
        return dict;
    }

    // Create a dict of word and key phrases
    // Allocate the key phrases based on the words
    static create_word_key_phrases_dict(title_words, key_phrases){
        let dict = {};
        // Initialise dict with an array
        for(const title_word of title_words){
            dict[title_word] = [];
        }
        // Add 'others'
        dict['others'] = [];

        // Match the key phrases with title words and assign key phrase
        for(const key_phrase of key_phrases) {
            let is_found = false;
            for(const title_word of title_words){
                if(key_phrase.toLowerCase().includes(title_word.toLowerCase())){
                    dict[title_word].push(key_phrase);
                    is_found = true;
                }
            }
            // If not found, assign to 'misc'
            if(!is_found){
                dict['others'].push(key_phrase);
            }
        }
        return dict;
    }

}


