'use restrict';
function CategoryView(_story){
    const story = _story;
    let container = $('<div class="text-left small"></div>');
    this.getContainer = function () {
        return container;
    };
    function _createUI(){
        let storyCategories = story['category'];
        let texts = storyCategories.map(c => "#"+ c.toLowerCase()).join(" ");

        let btn = $('<button class="btn btn-link btn-sm text-left" type="button">'+texts+'</button>');
        container.append(btn);
        // Collapse div
        let table = $('<table class="table table-sm collapse"><thead><th>Category</th><th>Terms</th></thead></table>');
        for (let storyCategory of storyCategories) {
            let terms = story[storyCategory + '_terms'];
            let row = $('<tr></tr>');
            row.append($('<td></td>').text(storyCategory));
            row.append($('<td></td>').text(terms.map(term => term[0]).join(', ')));
            table.append(row);
        }
        // Hide/show the table
        btn.on("click", function () {
            $(this).siblings('.collapse').collapse('toggle');
        });
        container.append(table);
    }
    _createUI();

}
