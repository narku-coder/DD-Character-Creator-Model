import 'dart:convert';
import 'package:flutter/material';
import 'package:http/http.dart' as http;

class ChatScreen extends StatefulWidget {
  const ChatScreen({Key? key}) : super(key: key);

  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final List<Map<String, dynamic>> _messages = [];
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  bool _isLoading = false;

  // Replace with your live FastAPI Render URL
  final String _apiUrl = "https://your-dnd-api.onrender.com/generate";

  Future<void> _sendMessage(String text) async {
    if (text.trim().isEmpty) return;

    _controller.clear();
    setState(() {
      _messages.add({"role": "user", "content": text, "isJson": false});
      _isLoading = true;
    });
    _scrollToBottom();

    try {
      final response = await http.post(
        Uri.parse(_apiUrl),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({"prompt": text}),
      );

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        setState(() {
          _messages.add({
            "role": "assistant",
            "content": response.body, // Store raw JSON string
            "isJson": true,
          });
        });
      } else {
        _showError("Error: Failed to fetch character data from server.");
      }
    } catch (e) {
      _showError("Network error: Could not reach backend.");
    } finally {
      setState(() => _isLoading = false);
      _scrollToBottom();
    }
  }

  void _showError(String errorMsg) {
    setState(() {
      _messages.add({"role": "assistant", "content": errorMsg, "isJson": false});
    });
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("D&D Character AI App"),
        backgroundColor: Colors.deepPurple,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              controller: _scrollController,
              padding: const EdgeInsets.all(12),
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                final bool isUser = msg["role"] == "user";

                return _buildChatBubble(isUser, msg);
              },
            ),
          ),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8.0),
              child: CircularProgressIndicator(color: Colors.deepPurple),
            ),
          _buildInputArea(),
        ],
      ),
    );
  }

  Widget _buildChatBubble(bool isUser, Map<String, dynamic> msg) {
    return Alignment(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 6),
        padding: const EdgeInsets.all(14),
        constraints: BoxConstraints(
          maxWidth: MediaQuery.of(context).size.width * 0.8,
        ),
        decoration: BoxDecoration(
          // Visually distinct themes for User vs Assistant
          color: isUser ? Colors.deepPurple : Colors.grey[200],
          borderRadius: BorderRadius.only(
            topLeft: const Radius.circular(16),
            topRight: const Radius.circular(16),
            bottomLeft: Radius.circular(isUser ? 16 : 0),
            bottomRight: Radius.circular(isUser ? 0 : 16),
          ),
        ),
        child: isUser 
          ? Text(
              msg["content"],
              style: const TextStyle(color: Colors.white, fontSize: 16),
            )
          : msg["isJson"]
              ? _buildCharacterSheetView(jsonDecode(msg["content"]))
              : Text(
                  msg["content"],
                  style: const TextStyle(color: Colors.black87, fontSize: 16),
                ),
      ),
    );
  }

  // Parses the AI model's enriched JSON block into a graphical character preview card
  Widget _buildCharacterSheetView(Map<String, dynamic> sheet) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              sheet["name"] ?? "Unknown Hero",
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 20, color: Colors.purple),
            ),
            IconButton(
              icon: const Icon(Icons.download, color: Colors.deepPurple),
              onPressed: () {
                // Hook up the PDF generation method here
              },
            )
          ],
        ),
        Text("Level ${sheet["level"]} ${sheet["species"]} ${sheet["class"]}"),
        if (sheet["subclass"] != null) Text("Subclass: ${sheet["subclass"]}"),
        const Divider(),
        const Text("Attributes:", style: TextStyle(fontWeight: FontWeight.bold)),
        Wrap(
          spacing: 8,
          children: (sheet["stats"] as Map<String, dynamic>).entries.map((stat) {
            return Chip(
              label: Text("${stat.key}: ${stat.value}"),
              backgroundColor: Colors.white,
            );
          }).toList(),
        ),
        const SizedBox(height: 8),
        if (sheet["enriched_features"] != null) ...[
          const Text("Features:", style: TextStyle(fontWeight: FontWeight.bold)),
          ...(sheet["enriched_features"] as List).map((feat) {
            return ExpansionTile(
              title: Text(feat["name"] ?? "", style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w500)),
              children: [
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(feat["description"] ?? "", style: TextStyle(color: Colors.grey[700])),
                )
              ],
            );
          }).toList()
        ]
      ],
    );
  }

  Widget _buildInputArea() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      color: Colors.white,
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _controller,
              decoration: const InputDecoration(
                hintText: "Ask for a character (e.g., Level 5 Rogue)...",
                border: OutlineInputBorder(borderRadius: BorderRadius.all(Radius.circular(24))),
                contentPadding: EdgeInsets.symmetric(horizontal: 16),
              ),
              onSubmitted: _sendMessage,
            ),
          ),
          const SizedBox(width: 8),
          IconButton(
            icon: const Icon(Icons.send, color: Colors.deepPurple),
            onPressed: () => _sendMessage(_controller.text),
          ),
        ],
      ),
    );
  }
}