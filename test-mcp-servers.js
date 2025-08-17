#!/usr/bin/env node

/**
 * MCP Server Test Script
 *
 * This script tests the availability and functionality of MCP servers,
 * particularly the sequential thinking server.
 */

const { spawn } = require('child_process');
const path = require('path');

console.log('🔧 Testing MCP Server Connectivity...\n');

async function testServer(serverName, command, args) {
    return new Promise((resolve) => {
        console.log(`📡 Testing ${serverName}...`);
        console.log(`   Command: ${command} ${args.join(' ')}`);

        const child = spawn(command, [...args, '--help'], {
            stdio: ['pipe', 'pipe', 'pipe'],
            shell: true
        });

        let stdout = '';
        let stderr = '';

        child.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        child.stderr.on('data', (data) => {
            stderr += data.toString();
        });

        child.on('close', (code) => {
            const success = code === 0 || stdout.includes('usage') || stdout.includes('help');

            console.log(`   Status: ${success ? '✅ Available' : '❌ Failed'}`);
            if (success && stdout) {
                console.log(`   Info: Server responds to --help`);
            } else if (stderr) {
                console.log(`   Error: ${stderr.trim().substring(0, 100)}`);
            }
            console.log('');

            resolve({
                name: serverName,
                success,
                stdout,
                stderr,
                exitCode: code
            });
        });

        child.on('error', (error) => {
            console.log(`   Status: ❌ Failed to start`);
            console.log(`   Error: ${error.message}`);
            console.log('');

            resolve({
                name: serverName,
                success: false,
                error: error.message
            });
        });

        // Timeout after 10 seconds
        setTimeout(() => {
            child.kill();
            console.log(`   Status: ⏰ Timeout`);
            console.log('');

            resolve({
                name: serverName,
                success: false,
                error: 'Timeout'
            });
        }, 10000);
    });
}

async function main() {
    const servers = [
        {
            name: 'Sequential Thinking (Official)',
            command: 'npx',
            args: ['@modelcontextprotocol/server-sequential-thinking']
        },
        {
            name: 'Sequential Thinking (Alternative)',
            command: 'npx',
            args: ['mcp-sequential-thinking']
        },
        {
            name: 'Playwright MCP',
            command: 'npx',
            args: ['-y', '@playwright/mcp']
        }
    ];

    const results = [];

    for (const server of servers) {
        const result = await testServer(server.name, server.command, server.args);
        results.push(result);
    }

    console.log('📊 Summary:');
    console.log('='.repeat(40));

    const successful = results.filter(r => r.success);
    const failed = results.filter(r => !r.success);

    console.log(`✅ Available servers: ${successful.length}`);
    successful.forEach(s => console.log(`   - ${s.name}`));

    if (failed.length > 0) {
        console.log(`❌ Failed servers: ${failed.length}`);
        failed.forEach(s => console.log(`   - ${s.name}: ${s.error || 'Unknown error'}`));
    }

    console.log('\n💡 Next steps:');
    if (successful.length > 0) {
        console.log('   1. Restart VS Code to reload MCP configuration');
        console.log('   2. Check that the Copilot MCP extension is enabled');
        console.log('   3. Try using the sequential thinking tool in Copilot chat');
    } else {
        console.log('   1. Install MCP servers: npm install -g @modelcontextprotocol/server-sequential-thinking');
        console.log('   2. Check Node.js version (requires Node.js 18+)');
        console.log('   3. Verify network connectivity for npm packages');
    }
}

main().catch(console.error);
