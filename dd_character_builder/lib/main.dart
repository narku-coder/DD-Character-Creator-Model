import 'package:flutter/material.dart';
import 'package:my_dnd_app/src/features/chat/chat_screen.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatelessWidget {
  const MainApp({super.key});
 
 @override
 Widget build(BuildContext context) {
    return MaterialApp(
      title: 'D&D AI Companion',
      theme: ThemeData(primarySwatch: Colors.deepPurple),
      home: const ChatScreen(), // Displays your chat UI on startup
    );
  }
}
