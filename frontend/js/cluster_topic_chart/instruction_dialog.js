// Display the instruction
function InstructionDialog(auto_open){
    let instruction_text = '1. <b>Mouse over a link </b> displays the similarity between two clusters and cluster topics<br>' +
            '2. <b>Clicking on a link</b> displays the articles relevant the topics of two clusters';

    $('#instruction').empty();
    $('#instruction').append($('<p>'+ instruction_text+'</p>'));
    $('#instruction').dialog({
        autoOpen: auto_open,
        modal: true,
        buttons: {
            Ok: function () {
                $(this).dialog("close");
            }
        }
    });
    $("#opener").on("click", function () {
        $("#instruction").dialog("open");
    });

}
