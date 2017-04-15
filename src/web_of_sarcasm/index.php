<?php
	require  'util.php';

	$errors = new Errors();

	if (isset($_POST['action'])) {

		$db = new DbAdapter();
		$action = $_POST['action'];

		if ($action == 'get_quiz') {
			//isset($_POST['user_id'])
			//isset($_POST['nonce'])
			$user_id = intval($_POST['user_id']);
			$nonce = $_POST['nonce'];
			$size = $_POST['size'];

			if ($db->validate_user($user_id, $nonce)) {
				$quiz = $db->getQuiz($size, 'poria-balanced');	
				
				$result_array = array(
					'type'=>'set_quiz',
					'quiz' =>$quiz
				);
				//$result_array = array('type'=>'set_quiz');
			} else {
				$errors->add("user not validated");
			}			

		} elseif ($action == 'register_user') {
			$name = $_POST['name'];
			$user = $db->createUser($name);
			$result_array = array(
				'type'=>'set_user',
				'id'=>$user['id'],
				'nonce'=>$user['nonce'],
				'name'=>$user['name']
			);			

		} elseif ($action == 'save_answer') {            
            $question_id = intval($_POST['question_id']);
            $user_id = intval($_POST['user_id']);
            $nonce = $_POST['nonce'];
            $answer = $_POST['answer'];

            if ($db->validate_user($user_id, $nonce)) {

                $db->setAnswer($question_id, $user_id, $answer);  
                
                $result_array = array(
                    'type'=>'saved'
                );
            } else {
                $errors->add("user not validated");
            }           
		} else {
			$errors->add("Request command not recognized");
		}

		if ($errors->has() || $db->errors->has()) {
			$all_errors = array_merge($errors->getAll(), $db->errors->getAll());
			$result_array = array(
				'type'=>'error',
				'errors'=>$all_errors
			);
		} 

		echo json_encode( $result_array );

		die();
	} 
?>

<!doctype html>
<!--[if lt IE 7]>      <html class="no-js lt-ie9 lt-ie8 lt-ie7" lang=""> <![endif]-->
<!--[if IE 7]>         <html class="no-js lt-ie9 lt-ie8" lang=""> <![endif]-->
<!--[if IE 8]>         <html class="no-js lt-ie9" lang=""> <![endif]-->
<!--[if gt IE 8]><!--> <html class="no-js" lang=""> <!--<![endif]-->
    <head>
        <meta charset="utf-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">
        <title></title>
        <meta name="description" content="">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="apple-touch-icon" href="apple-touch-icon.png">

        <link rel="stylesheet" href="css/normalize.min.css">
        <link rel="stylesheet" href="css/main.css">

        <script src="js/vendor/modernizr-2.8.3-respond-1.4.2.min.js"></script>
    </head>
    <body>
        <!--[if lt IE 8]>
            <p class="browserupgrade">You are using an <strong>outdated</strong> browser. Please <a href="http://browsehappy.com/">upgrade your browser</a> to improve your experience.</p>
        <![endif]-->

        <div class="header-container">
            <header class="wrapper clearfix">
                <h1 class="title">Sarcasm quiz!</h1>
                <nav id='navbar' hidden=true>
                    <ul>
                        <li><a href="#">Hard quiz</a></li>
                        <li><a href="#">Easy quiz</a></li>
                    </ul>
                </nav>
            </header>
        </div>

        <div class="main-container">
            <div class="main wrapper clearfix">

            	<article id='main-content'>
            	    <header id='content-header' hidden=true>
            	        <h3>Please provide your name</h3>
            	        <form id='name-form'>
            	        	<input type='text' name='name'><br>
            	        	<input type='submit' value='Submit' id='submit-name'>
            	        </form>
            	    </header>        	    
            	</article>

            	<!--
                <article>
                    <header>
                        <h1>Sarcasm quiz!</h1>
                        <p>Below you will be presented with a random selection of tweets. Please guess if you think thay were tagged with #sarcasm or not!</p>
                    </header>
                    
                    <section>
                        <h2>1.</h2>
                        <p>Omg sarcastic sarcastic sarcastic</p>
                    </section>

                  
                    <footer>
                        <h3>Quiz info</h3>
                        <p>The easy quiz contains an even amount of sarcastic and none-sarcastic tweets. The hard quiz has a ratio of 1 to 3 betwean sarcastic and none-sarcastic tweets</p>
                    </footer>
                    
                </article>
				-->
                <aside id='aside' hidden=true>
                	<div id='status' hidden=true></div>
                	<div id='score' hidden=true>
	                    <h3>Your score:</h3>
	                    <p>Accuracy:</p>
	                    <p>F1 score:</p>
	                </div>
                </aside>

            </div> <!-- #main -->
        </div> <!-- #main-container -->

        <div class="footer-container">
            <footer class="wrapper">
                <h3>footer</h3>
            </footer>
        </div>

        <script src="js/vendor/jquery-1.11.2.js"></script>
        <script src="js/main.js"></script>

        <!--
        <script src="//ajax.googleapis.com/ajax/libs/jquery/1.11.2/jquery.js"></script>
        <script>window.jQuery || document.write('<script src="js/vendor/jquery-1.11.2.js"><\/script>')</script>

        

         Google Analytics: change UA-XXXXX-X to be your site's ID. 
        <script>
            (function(b,o,i,l,e,r){b.GoogleAnalyticsObject=l;b[l]||(b[l]=
            function(){(b[l].q=b[l].q||[]).push(arguments)});b[l].l=+new Date;
            e=o.createElement(i);r=o.getElementsByTagName(i)[0];
            e.src='//www.google-analytics.com/analytics.js';
            r.parentNode.insertBefore(e,r)}(window,document,'script','ga'));
            ga('create','UA-XXXXX-X','auto');ga('send','pageview');
        </script>
        -->
    </body>
</html>

