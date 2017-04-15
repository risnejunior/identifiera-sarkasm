var ajaxurl = "index.php"

// js format implementation
if (!String.prototype.format) {
  String.prototype.format = function() {
    var args = arguments;
    return this.replace(/{(\d+)}/g, function(match, number) { 
      return typeof args[number] != 'undefined'
        ? args[number]
        : match
      ;
    });
  };
}

var entityMap = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#39;',
  '/': '&#x2F;',
  '`': '&#x60;',
  '=': '&#x3D;'
};

// solution from mustache.js
function escapeHtml (string) {
  return String(string).replace(/[&<>"'`=\/]/g, function (s) {
    return entityMap[s];
  });
}

$( document ).ready(function() {
	var user_id = null 

	//user unknown
	if (localStorage.getItem('user_id') === null &&
		localStorage.getItem('nonce') === null ) {
		
		$("#content-header").show();

	//user known
	} else {
		set_for_quiz();
		get_quiz(5);
	}
	

	
});

$("#name-form").submit(function(event) {
	event.preventDefault();

	var message = {
		'name': $( "input[name*='name']" ).first().val(),
		'action': 'register_user'
	};

	get_ajax_data(message)
});

function set_for_quiz() {
	console.log("set for quiz")
	jQuery("#aside, #navbar").show();
	jQuery("#content-header").hide();
	jQuery("#status").html("Welcome " + localStorage['name']).show();
}

function get_ajax_data(message) {
	var settings = {
	     "url": ajaxurl,
	     "timeout": 5000,
	     "type": "post",
	     "dataType": "json",
	     "data": message,
	     "complete": complete,
	     "error": error,
	     "success": success,
	     "context": this
	};
	 
	 //ajax call bound to deferred
	jQuery.ajax( settings );

	function success( data, textStatus, jqXHR ) {
		//serverside error 
	    if ( !data.type || "error" === data.type ) {
	        var err_message = "An error occurred!";
	        if ( 'errors' in data ) {
	            err_message = "Error: " + data.errors.join(',');
	        }
	        jQuery("#status").html( err_message ).show();
	    
	    //server side success
	    } else if (data.type === 'set_quiz') {	    		    
	    	localStorage['quiz'] = JSON.stringify(data.quiz);
	    	add_quiz_html();
	    } else if (data.type === 'set_user') {
	    	set_for_quiz();
	    	localStorage['user_id'] = data.id;
	    	localStorage['nonce'] = data.nonce;
	    	localStorage['name'] = data.name;
		} 
	}

	function error( jqXHR, textStatus, errorThrown ) {
		jQuery("#status").html( "An error occurred!" ).show();
	}

	function complete( jqXHR, textStatus ) {
		jQuery( this )
		    .parent()
		    .siblings( ".animation-container" )
		    .removeClass( "working-animation" );
	}
}

function get_quiz(size, dataset) {
	
	var message = {
		'action': 'get_quiz',
		'user_id': parseInt(localStorage['user_id']),
		'nonce': localStorage['nonce'],
		'size': size
	};

	var data = get_ajax_data(message)
}

function add_quiz_html() {
	var quiz = JSON.parse(localStorage['quiz']);
	for (var i = 0; i < quiz.length; i++) {		
		question = quiz[i];
		console.log(question.sample_text);

		var sectionElement = $("<section></section>")
			.attr("id", question.id)
			.text(question.sample_text)
			.css("background", "#f8ede8")
			.css("margin", "5px")
			.css("padding", "5px")
			.css("border-radius", "6px");

		var buttonPart = $("<div></div>")
			.addClass("button-part")
			.append("<button class='pos button'>Sarcastic</button>")
			.append("<button class='neg button'>Other</button>");

		sectionElement.append(buttonPart);
		$("#main-content").append(sectionElement);
	}

	$("button", ".button-part").click(function(event) {
		var question_id = $(this).parents("section").first().attr("id");		
		var isPos = $(this).hasClass("pos");
		
		if ( !$(this).hasClass('selected') ) {
			$(this).addClass('selected');
		}

		$(this).siblings('.button:first').removeClass('selected');

		set_answer(question_id, isPos);
	});
}

function set_answer(question_id, isPos) {
	console.log("Set answer: " + question_id + " " + isPos);
	user_id = localStorage.getItem('user_id')
	nonce = localStorage.getItem('nonce')
	var message = {
		'action': 'save_answer',
		'nonce': nonce,
		'question_id': parseInt(question_id),
		'user_id': parseInt(user_id),		
		'answer': isPos
	};

	get_ajax_data(message)
}