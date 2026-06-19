import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl = "https://your-dnd-api.onrender.com";

  // Sends the chat prompt and the session token to the backend
  Future<Map<String, dynamic>?> sendChatMessage(String sessionId, String prompt) async {
    final url = Uri.parse("$baseUrl/generate");
    
    try {
      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "session_id": sessionId,
          "prompt": prompt,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        print("Server Error: ${response.statusCode}");
        return null;
      }
    } catch (e) {
      print("Network Connection Exception: $e");
      return null;
    }
  }
}