package server

import (
	"fmt"
	"log/slog"
	"strings"
	"text/template"
	"text/template/parse"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
)

// isResponseNode checks if the node contains .Response
func isResponseNode(node *parse.ActionNode) bool {
	for _, cmd := range node.Pipe.Cmds {
		for _, arg := range cmd.Args {
			if fieldNode, ok := arg.(*parse.FieldNode); ok && len(fieldNode.Ident) > 0 {
				if fieldNode.Ident[0] == "Response" {
					return true
				}
			}
		}
	}
	return false
}

// formatTemplateForResponse formats the template AST to:
// 1. remove all nodes after the first .Response (if cut=true)
// 2. add a .Response node to the end if it doesn't exist
// TODO(jmorganca): this should recursively cut the template before the first .Response
func formatTemplateForResponse(tmpl *template.Template, cut bool) {
	var found bool
	for i, node := range tmpl.Tree.Root.Nodes {
		if actionNode, ok := node.(*parse.ActionNode); ok {
			if isResponseNode(actionNode) {
				found = true
				if cut {
					tmpl.Tree.Root.Nodes = tmpl.Tree.Root.Nodes[:i+1]
					break
				}
			}
		}
	}

	if !found {
		// add the resposne node if it doesn't exist
		responseFieldNode := &parse.FieldNode{NodeType: parse.NodeField, Ident: []string{"Response"}}
		responsePipeNode := &parse.PipeNode{NodeType: parse.NodePipe, Cmds: []*parse.CommandNode{{NodeType: parse.NodeCommand, Args: []parse.Node{responseFieldNode}}}}
		responseActionNode := &parse.ActionNode{NodeType: parse.NodeAction, Pipe: responsePipeNode}
		tmpl.Tree.Root.Nodes = append(tmpl.Tree.Root.Nodes, responseActionNode)
	}
}

func Prompt(tmpl, system, prompt, response string, cut bool) (string, error) {
	parsed, err := template.New("").Option("missingkey=zero").Parse(tmpl)
	if err != nil {
		return "", err
	}

	formatTemplateForResponse(parsed, cut)

	vars := map[string]any{
		"System":   system,
		"Prompt":   prompt,
		"Response": response,
	}

	var sb strings.Builder
	if err := parsed.Execute(&sb, vars); err != nil {
		return "", err
	}

	return sb.String(), nil
}

func countTokens(tmpl string, system string, prompt string, response string, encode func(string) ([]int, error)) (int, error) {
	rendered, err := Prompt(tmpl, system, prompt, response, false)
	if err != nil {
		return 0, err
	}

	tokens, err := encode(rendered)
	if err != nil {
		slog.Error("failed to encode prompt", "err", err)
		return 0, err
	}

	return len(tokens), err
}

// ChatPrompt builds up a prompt from a series of messages, truncating based on context window size
func ChatPrompt(tmpl string, system string, messages []api.Message, window int, encode func(string) ([]int, error)) (string, error) {
	type prompt struct {
		System   string
		system   bool
		Prompt   string
		prompt   bool
		Response string
		response bool

		Images []llm.ImageData

		tokens int
	}

	var prompts []prompt

	var p prompt

	// Set the first system prompt to the model's system prompt
	p.System = loaded.Model.System
	p.system = true

	// iterate through messages to build up prompts
	var images int
	for _, msg := range messages {
		switch strings.ToLower(msg.Role) {
		case "system":
			if p.system || p.prompt || p.response {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.System = msg.Content
			p.system = true
		case "user":
			if p.prompt || p.response {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.Prompt = msg.Content
			p.prompt = true

			for i := range msg.Images {
				p.Prompt += fmt.Sprintf(" [img-%d]", images)
				p.Images = append(p.Images, llm.ImageData{
					ID:   images,
					Data: msg.Images[i],
				})
				images += 1
			}
		case "assistant":
			if p.response {
				prompts = append(prompts, p)
				p = prompt{}
			}

			p.Response = msg.Content
			p.response = true
		default:
			return "", fmt.Errorf("invalid role: %s, role must be one of [system, user, assistant]", msg.Role)
		}
	}

	// add final prompt
	if p.system || p.prompt || p.response {
		prompts = append(prompts, p)
	}

	// calculate token lengths for each prompt, estimating 768 for images
	for _, p := range prompts {
		tokens, err := countTokens(tmpl, p.System, p.Prompt, p.Response, encode)
		if err != nil {
			return "", err
		}

		p.tokens = tokens + len(p.Images)*768
	}

	// truncate images and prompts starting from the beginning of the list
	// until either one prompt remains or the total tokens fits the context window
	// TODO (jmorganca): this doesn't account for the context window room required
	for {
		// estimate the total tokens required for the prompt
		var total int
		for _, p := range prompts {
			total += p.tokens
		}

		if total <= window {
			break
		}

		if len(prompts[0].Images) > 1 {
			img := prompts[0].Images[0]
			slog.Debug("context window too long, removing image", "id", img.ID, "tokens", total, "window", window)
			prompts[0].Images = prompts[0].Images[1:]
			prompts[0].tokens -= 768
			continue
		}

		if len(prompts) > 1 {
			slog.Debug("context window too long, removing prompt", "prompt", prompts[0], "tokens", total, "window", window)
			system := prompts[0].System
			prompts = prompts[1:]

			if system == "" {
				continue
			}

			// bring along system prompt and recalulate tokens for it
			if prompts[0].System == "" {
				prompts[0].System = system

				tokens, err := countTokens(tmpl, prompts[0].System, prompts[0].Prompt, prompts[0].Response, encode)
				if err != nil {
					return "", err
				}

				prompts[0].tokens = tokens + len(prompts[0].Images)*768
			}
		}

		// stop truncating if we can't remove any more prompts
		break
	}

	var sb strings.Builder
	for _, p := range prompts {
		rendered, err := Prompt(tmpl, p.System, p.Prompt, p.Response, true)
		if err != nil {
			return "", err
		}
		sb.WriteString(rendered)
	}

	return sb.String(), nil
}
