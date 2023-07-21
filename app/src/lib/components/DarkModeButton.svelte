<script>
	import { MoonRegular, SunRegular } from 'svelte-awesome-icons';
	import { darkMode } from '$lib/utils';
	const toggle = () => darkMode.update((x) => !x);
</script>

<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div on:click={toggle}>
	{#if $darkMode}
		<SunRegular />
	{:else}
		<MoonRegular />
	{/if}
</div>
