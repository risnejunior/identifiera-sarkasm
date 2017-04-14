<?php

//get user
//set user
//get quiz
//set answers

class Errors {
	protected $errors = array();

	public function add($error) {
		array_push($this->errors, $error);
	}

	public function getAll() {
		return $this->errors;
	}

	public function has() {
		$hasErrors = false;
		if ($this->errors) {
			$hasErrors = true;
		}

		return $hasErrors;
	}
}

class DbAdapter {
	const DB_SERVER = "localhost";
	const DB_USER = "root";
	const DB_PASSWORD = "";
	const DB_NAME = "sarcasm_db";

	public $errors;
	private $conn;

	function __construct() {
		$this->errors = new Errors();
		
		// Create connection
		$this->conn = new mysqli(
			self::DB_SERVER, 
			self::DB_USER, 
			self::DB_PASSWORD, 
			self::DB_NAME
		);

		// Check connection
		if ($this->conn->connect_error) {
		    //die("Connection failed: " . $conn->connect_error);
		    $this->errors->add($conn->connect_error);
		} 		

		$this->conn->set_charset("utf8");
	}

	function __destruct() {
		$this->conn->close();
	}

	public function validate_user($user_id, $nonce) {
		$user = $this->getUser($user_id);
		$validated = false;
		
		if ($user && $user['nonce'] == $nonce) {
			$validated = true;
		} else {
			$this->errors->add('User not validated');
		}

		return $validated;
	}

	public function getQuiz($size, $dataset) {
		$sql = "
		SELECT id, sample_text
		FROM samples AS r1 
		JOIN (SELECT CEIL(RAND() * (SELECT MAX(id)
			FROM samples)) AS ids) AS r2
		 WHERE r1.id >= r2.ids
		 AND dataset = '{$dataset}'
		 ORDER BY r1.id ASC
		 LIMIT {$size}
		 ";
		 $rows = $this->execQuery($sql);

		 return $rows;
	}

	public function getUser($user_id) {
		$user = NULL;
		if (!is_int($user_id)) {
			$this->errors->add('User id has to be an int');
		} else {

			$sql = "SELECT user_id, name, nonce FROM users WHERE user_id = {$user_id}";
			$rows = $this->execQuery($sql);
			$user = NULL;

			if (count($rows) == 1) {
				$user = $rows[0];
			} else {
				$this->errors->add('User not found');
			}
		}

		return $user;
	}

	public function createUser($user_name) {
		$nonce = self::createNonce();
		$ip = $_SERVER['REMOTE_ADDR'];
		$name = $this->conn->real_escape_string($user_name);
		$sql = "INSERT INTO users (name, ip, nonce) VALUES ('{$name}', '{$ip}', '{$nonce}');";
		$this->execQuery($sql);
		$insert_id = $this->conn->insert_id;
		if ($this->conn->connect_error) {
		    //die("Connection failed: " . $conn->connect_error);
		    $this->errors->add($conn->connect_error);
		} 

		$user = array(
			'id'=>$insert_id,
			'name'=>$name,
			'nonce'=>$nonce
		);

		return $user;
	}


	private function execQuery($sql) {
		$result = $this->conn->query($sql);
		$this->conn->store_result();
		$rows = NULL;
		if (isset($result->num_rows)) {
			$rows = array();
			$i = 0;
		    while($row = $result->fetch_assoc()) {		    	
		        $rows[$i] = $row;
		        $i++;
		    }
		} 		

		return $rows;
	}

	public static function createNonce() {
	    $id = 1; // replace with actual username
	    $nonce = hash('md4', self::randomString(10)); //not 'secure' but fast

	    return $nonce;
	}


	private static function randomString($length = 10) {
	    $characters = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ';
	    $charactersLength = strlen($characters);
	    $randomString = '';
	    for ($i = 0; $i < $length; $i++) {
	        $randomString .= $characters[rand(0, $charactersLength - 1)];
	    }
	    return $randomString;
	}
}

?>